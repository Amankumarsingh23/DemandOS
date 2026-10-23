"""One native origin, six fixed series, existing decision/simulation engines."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import pandas as pd

from demandos.serving.core import atomic_json, digest, require, read_json, file_hash
from demandos.serving.inputs import load, verify_sources
from demandos.serving.features import contract, extract, parity, KEY
from demandos.serving.bootstrap import initialize, APPROVED
from demandos.serving.prediction import predict
from demandos.serving.decisions import frozen_inputs, plan, downstream, resource_record, target_vector
from demandos.serving.monitoring import mature_labels
from demandos.causal.simulation import intervention
from demandos.risk.scenarios import Closure
from demandos.reconciliation.pipeline import peak_memory
from .config import CONFIG, LABELS, code_identity, prerequisites, table_label
from .evidence import collect, metrics
from .faults import exercise
from .audit import audit, prefix, trajectory_checks, storage_checks
from .publication import publish


def select(metadata):
    # Cohort membership was frozen in Season 16 from earlier history. No y is read.
    selected = metadata.sort_values("series_id").groupby("cohort", sort=True).head(CONFIG["per_cohort"])
    require(len(selected) == 6 and selected.cohort.nunique() == 2, "CAPSTONE_FIXED_COHORT")
    return selected.sort_values("series_id").reset_index(drop=True)


def capacities(metadata, resources):
    # Worst-case all horizon supplier entitlements land in this store, with no sales.
    # Conservative hard ceiling, NOT an estimated or optimized Walmart capacity.
    return {store: float(13*g.volume.sum()+sum(resources.suppliers.values())*g.volume.max())
            for store, g in metadata.groupby("store_id")}


def run(root):
    root = Path(root).resolve()
    ready = prerequisites(root)
    require(ready["ready"], "MISSING_PREREQUISITES", ready["missing"])
    started = perf_counter()
    data = load(root)
    metadata = select(data.metadata)
    snapshot = data.snapshot.merge(metadata[KEY], on=KEY, validate="many_to_one")
    feature_contract = contract(root, data.feature_config, data.mapping)
    contract_digest = digest(feature_contract)
    features, source_digest = extract(snapshot, data.calendar, metadata, CONFIG["origin"], data.feature_config, data.mapping)
    changed = snapshot.copy()
    future = changed.date.gt(CONFIG["origin"])
    changed.loc[future, "units_sold"] += 100000
    changed.loc[future, "sell_price"] = 99999.
    other, other_digest = extract(changed, data.calendar, metadata, CONFIG["origin"], data.feature_config, data.mapping)
    parity(features, other)
    require(source_digest == other_digest, "FUTURE_SALES_PRICE_DEPENDENCE")
    basis, scenarios, resources, _ = frozen_inputs(snapshot, metadata, CONFIG["origin"], data.settings)
    storage = capacities(metadata, resources)
    evidence = collect(root, digest(data.source_hashes))
    prior_run = root/read_json(root/"data/serving/m5_h28/latest_verified.json")["path"]
    prior_registry = {p.relative_to(root).as_posix():file_hash(p) for p in sorted((prior_run/"registry").rglob("*.json"))}
    base = root/"data/capstone/m5_h28"

    def build(stage):
        registry, _ = initialize(data, stage/"registry", contract_digest)
        forecasts, _ = predict(registry, APPROVED, features, snapshot, contract_digest, approved=True)
        decisions, solver = plan(forecasts, metadata, data.settings, scenarios, resources)
        origin = pd.Timestamp(CONFIG["origin"])
        dates = pd.date_range(origin+pd.Timedelta(days=1), periods=28)
        actual = np.stack([snapshot[snapshot.store_id.eq(m.store_id) & snapshot.item_id.eq(m.item_id)]
                           .set_index("date").loc[dates, "units_sold"].to_numpy(int) for m in metadata.itertuples()])
        main, _ = downstream(actual, origin, metadata, forecasts, scenarios, data.settings, basis)
        modified = actual.copy(); modified[0, CONFIG["perturbation_day"]-1] += 1000
        perturbed, _ = downstream(modified, origin, metadata, forecasts, scenarios, data.settings, basis)
        prefix(main, perturbed, CONFIG["perturbation_day"])
        closure = intervention(actual, origin, metadata, target_vector(forecasts, metadata), scenarios,
                               {**data.settings, "resource_targets":basis}, action=0, closure=Closure(8,14))
        frames = dict(metadata=metadata, snapshot=snapshot, features=features, forecasts=forecasts, decisions=decisions)
        state_metrics = None
        for variant, result in [("", main), ("perturbed_", perturbed), ("closure_", closure)]:
            measured = trajectory_checks(result, len(metadata))
            storage_checks(result["daily"], decisions, metadata, storage)
            if not variant:
                state_metrics = measured
            for name in ["daily", "events", "allocations", "resources", "solver"]:
                frame = result[name].copy()
                # Wall-clock solver timings/gaps are diagnostics, not action identity.
                if name == "solver":
                    frame = frame[["day_index", "status"]]
                frames[variant+name] = frame
        maturity = mature_labels(forecasts, snapshot, CONFIG["monitoring_date"], lag_days=CONFIG["label_lag_days"])
        frames["maturity"] = maturity
        faults = exercise(data, features, decisions, registry, contract_digest)
        model = registry.verify(APPROVED, contract_digest=contract_digest, origin=origin, approved=True)
        manifest = dict(schema_version=1, config=CONFIG, labels=LABELS, code=code_identity(root),
            table_labels={name:table_label(name) for name in frames},
            packages=ready["packages"], sources=data.source_hashes, models=data.model_hashes, model=model,
            prior_registry=prior_registry,
            warehouse={k:str(v) for k,v in data.warehouse_totals.items()},
            cohort=metadata[[*KEY,"cohort","series_id"]].to_dict("records"), eligible_snapshot_digest=source_digest,
            feature_contract_digest=contract_digest, settings=data.settings, resource_targets=basis.tolist(),
            origin_resources=resource_record(resources), storage_capacities=storage,
            simulation=dict(on_hand=10, prepaid_pipeline=3, lead_days=7, review_days=7, horizon=28,
                holding=.2,lost_penalty=40,backlog_daily=2,purchase=60,activation=5,
                chronology="receive, serve existing backlog, serve daily observed-sales demand proxy, observe stock position, approve shared orders",
                partial_supply="not modeled; whole-order receiving closure only; quantity and identity preserved",
                opening_plan="day-zero noncommitted preview; sequential controller re-solves after daily served sales; separate ledgers, no double cash booking"))
        summary = dict(forecast_metrics=metrics(maturity), simulation_metrics=state_metrics,
            label=LABELS["inventory"], forecast_label=LABELS["forecasts"], series=6, origin=CONFIG["origin"],
            forecast_rows=len(forecasts), state_rows=len(main["daily"]), simulated_paths=3,
            original_sources_and_models_unchanged=True, future_sales_and_prices_excluded=True,
            frozen_scenario_seed=42, loaded_native_models=["p10", "p50", "p90"], fit_cutoff=model["training_cutoff"],
            independent_storage="conservative assumed whole-horizon occupancy ceiling; not an active storage optimization study",
            warnings=["Recorded sales proxy, not verified latent demand.", "Costs, initial inventory, supplier entitlements, storage and arrival availability are assumed.",
                      "Whole-order delays only; no partial-supply or live dispatch integration.", "Six selected series and one origin; no population-wide or causal benefit claim."])
        (stage/"tables").mkdir()
        for name, frame in frames.items():
            frame.to_parquet(stage/"tables"/(name+".parquet"), index=False)
        for name, value in [("manifest",manifest),("feature_contract",feature_contract),("evidence",evidence),("summary",summary),("faults",faults)]:
            atomic_json(stage/(name+".json"), value)
        verified = audit(stage, data=data, sealed=False)
        atomic_json(stage/"verification.json", verified)
        atomic_json(stage/"runtime.json", dict(seconds=perf_counter()-started, peak_memory_mib=peak_memory(),
            scope="loading, source hashing, native inference, three bounded simulator paths, scientific reconciliation and staged independent audit; no retraining"))
        (stage/"capstone_report.md").write_text(report(summary, verified), encoding="utf-8")

    output, reused = publish(base, build, lambda p, sealed=True: audit(p, data=data, sealed=sealed))
    verify_sources(data)
    verified = audit(output, data=data)
    # Invocation observations can change on replay, so they live outside immutable bundles.
    invocation = dict(run_id=verified["run_id"], path=output.relative_to(root).as_posix(), reused=reused,
                      no_duplicate_actions=True, seconds=perf_counter()-started, peak_memory_mib=peak_memory(), verification=verified)
    atomic_json(base/"last_invocation.json", invocation)
    return invocation


def report(summary, verified):
    f = summary["forecast_metrics"]; s = summary["simulation_metrics"]
    return f"""# DemandOS Season 24 bounded local replay

Six predetermined series, origin {summary['origin']}, 28 days, native S16 P10/P50/P90 models fit through {summary['fit_cutoff']}.
Run `{verified['run_id']}`. Forecasts: {summary['forecast_rows']}; primary state rows: {summary['state_rows']}.

Historical recorded-sales MAE {f['mae']:.6f}, WAPE {f['wape']:.4%}, mean pinball {f['pinball']:.6f}, P10–P90 coverage {f['coverage']:.4%}.
SIMULATED inventory cost {s['cost']:.2f}, immediate fill {s['fill']:.4%}, procurement/activation committed {s['budget_used']:.2f} of {s['budget_initial']:.2f}.

The preview allocation and complete-policy simulation are separate experiments. Only the latter books sequential orders.
Resources persist across the horizon. Physical storage uses an explicit conservative assumed ceiling; flow-volume quotas are recorded separately.
Primary, day-14 demand perturbation and receiving-closure paths are audited. No original source/checkpoint or Season 23 registry is modified.
Five isolated fault exercises are in `faults.json`; independent arithmetic and invariants are in `verification.json`.
Cross-season measurements are recomputed from saved tables in `evidence.json`, not all retrained or fully rerun.

{chr(10).join('- '+x for x in summary['warnings'])}
"""
