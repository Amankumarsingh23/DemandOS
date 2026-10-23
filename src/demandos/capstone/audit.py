"""Read records and recompute correctness; primary 'passed' flags are not inputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from demandos.serving.core import digest, file_hash, read_json, require
from demandos.serving.features import CASE, KEY, extract, parity
from demandos.serving.decisions import audit_simulation, validate_decisions
from .config import CONFIG, LABELS, run_identity, code_identity, prerequisites, table_label
from .evidence import metrics
from .publication import check_seal


def prefix(first, second, day):
    """Compare every recorded field before changed demand, including shared state."""
    for name in ["daily", "allocations", "resources"]:
        a = first[name][first[name].day_index.lt(day)].reset_index(drop=True)
        b = second[name][second[name].day_index.lt(day)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-8, rtol=0)


def forecast_checks(forecasts, metadata, origin, entry, contract_digest):
    origin = pd.Timestamp(origin)
    require(len(forecasts) == len(metadata)*28 and not forecasts.duplicated(CASE).any(), "CAPSTONE_FORECAST_COVERAGE")
    require(forecasts.origin_date.eq(origin).all(), "CAPSTONE_ORIGIN")
    expected = {(s, i, h) for s, i in metadata[KEY].itertuples(index=False, name=None) for h in range(1,29)}
    require(set(forecasts[[*KEY, "horizon"]].itertuples(index=False, name=None)) == expected, "CAPSTONE_FORECAST_KEYS")
    require(forecasts.target_date.eq(origin+pd.to_timedelta(forecasts.horizon, unit="D")).all(), "CAPSTONE_TARGET_DATES")
    values = forecasts[["q10", "q50", "q90", "prediction"]].to_numpy(float)
    require(np.isfinite(values).all() and (values >= 0).all() and forecasts.q10.le(forecasts.q50).all()
            and forecasts.q50.le(forecasts.q90).all() and forecasts.prediction.eq(forecasts.q50).all(), "CAPSTONE_QUANTILES")
    require(pd.Timestamp(entry["training_cutoff"]) <= origin and pd.Timestamp(entry["validation_window"][1]) <= origin, "CAPSTONE_MODEL_CUTOFF")
    require(forecasts.model_digest.eq(entry["artifact_digest"]).all() and forecasts.model_version.eq(entry["model_version"]).all()
            and forecasts.feature_contract_digest.eq(contract_digest).all(), "CAPSTONE_MODEL_LINEAGE")


def storage_checks(daily, decisions, metadata, capacities):
    require(all(np.isfinite(x) and x >= 0 for x in capacities.values()), "STORAGE_LIMIT")
    # Opening inventory/pipeline plus all preview orders is a conservative occupancy bound.
    for store, g in decisions.groupby("store_id"):
        require(float(((13+g.approved_quantity)*g.volume).sum()) <= capacities[store]+1e-8, "STORAGE_PREVIEW")
    x = daily.merge(metadata[["series_id", "volume"]], on="series_id", validate="many_to_one")
    for (store, _), g in x.groupby(["store_id", "day_index"]):
        require(float(((g.opening_stock+g.arrivals)*g.volume).sum()) <= capacities[store]+1e-8, "STORAGE_OCCUPANCY")


def trajectory_checks(result, expected_series):
    d = result["daily"].sort_values(["series_id", "day_index"])
    require(len(d) == expected_series*28 and not d.duplicated(["series_id", "day_index"]).any(), "STATE_COVERAGE")
    require(not result["allocations"].order_request_id.duplicated().any(), "DUPLICATE_SIMULATED_ORDER")
    for _, g in d.groupby("series_id"):
        require(g.day_index.tolist() == list(range(1,29)), "STATE_DAY_KEYS")
        for opening, ending in [("opening_stock", "ending_stock"), ("opening_on_order", "ending_on_order"), ("opening_backlog", "ending_backlog")]:
            require(np.array_equal(g[opening].to_numpy()[1:], g[ending].to_numpy()[:-1]), "STATE_CONTINUITY", opening)
        require((g.opening_backlog+g.new_backlog-g.old_backlog_fulfilled == g.ending_backlog).all(), "BACKLOG_CONSERVATION")
    require(result["allocations"].activation.eq(result["allocations"].approved.gt(0).astype(int)).all(), "ORDER_ACTIVATION")
    events = result["events"]
    received = events[events.event.eq("received")]
    require(received.event_day.eq(received.arrival_day).all(), "RECEIPT_DATE")
    if "previous_arrival_day" in events:
        delayed = events[events.event.eq("retimed")]
        require(delayed.arrival_day.ge(delayed.previous_arrival_day).all(), "DELAY_REVERSES_TIME")
    return audit_simulation(result)


def audit(directory, *, data=None, sealed=True):
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json")
    identifier = run_identity(manifest)
    if sealed:
        require(check_seal(directory) == identifier, "CAPSTONE_IDENTITY")
    require(manifest["config"] == CONFIG and manifest["labels"] == LABELS, "SCIENTIFIC_LABEL_OR_CONFIG")
    frames = {p.stem: pd.read_parquet(p) for p in (directory/"tables").glob("*.parquet")}
    require(manifest["table_labels"] == {name:table_label(name) for name in frames}, "TABLE_SCIENTIFIC_LABELS")
    forecast_checks(frames["forecasts"], frames["metadata"], CONFIG["origin"], manifest["model"], manifest["feature_contract_digest"])
    validate_decisions(frames["decisions"])
    main = {k: frames[k] for k in ["daily", "events", "allocations", "resources", "solver"]}
    state_metrics = trajectory_checks(main, len(frames["metadata"]))
    storage_checks(main["daily"], frames["decisions"], frames["metadata"], manifest["storage_capacities"])
    for variant in ["perturbed", "closure"]:
        other = {k: frames[variant+"_"+k] for k in main}
        trajectory_checks(other, len(frames["metadata"]))
        storage_checks(other["daily"], frames["decisions"], frames["metadata"], manifest["storage_capacities"])
        if variant == "perturbed":
            prefix(main, other, CONFIG["perturbation_day"])
            key = ["series_id","day_index"]
            comparison = main["daily"].merge(other["daily"], on=key, suffixes=("_base","_changed"), validate="one_to_one")
            delta = comparison.demand_proxy_changed-comparison.demand_proxy_base
            expected_change = comparison.series_id.eq(frames["metadata"].series_id.iloc[0]) & comparison.day_index.eq(CONFIG["perturbation_day"])
            require(np.array_equal(delta.to_numpy(), np.where(expected_change,1000,0)), "PERTURBATION_DESIGN")
        else:
            require(other["events"].event.eq("retimed").any(), "CLOSURE_NOT_EXERCISED")
            require(np.array_equal(other["daily"].demand_proxy, main["daily"].demand_proxy), "CLOSURE_CHANGED_DEMAND")
    # Reconcile independently against recorded historical source target dates.
    source = frames["snapshot"]
    expected = source.rename(columns={"date":"target_date", "units_sold":"actual"})[[*KEY,"target_date","actual"]]
    scored = frames["forecasts"].merge(expected, on=[*KEY,"target_date"], how="left", validate="one_to_one")
    require(scored.actual.notna().all(), "CAPSTONE_LABEL_COVERAGE")
    observed = frames["daily"].merge(expected, left_on=[*KEY,"simulation_date"], right_on=[*KEY,"target_date"], validate="one_to_one")
    require(observed.demand_proxy.eq(observed.actual).all(), "DEMAND_SOURCE_ALIGNMENT")
    maturity = frames["maturity"]
    require(maturity.assumed_available_at.eq(maturity.target_date+pd.Timedelta(days=CONFIG["label_lag_days"])).all()
            and maturity.mature.eq(maturity.assumed_available_at.le(pd.Timestamp(CONFIG["monitoring_date"]))).all(), "CAPSTONE_LABEL_MATURITY")
    require(maturity.loc[~maturity.mature,"actual"].isna().all(), "IMMATURE_LABEL_DISCLOSURE")
    pd.testing.assert_frame_equal(maturity[CASE+["actual"]].reset_index(drop=True), scored[CASE+["actual"]].reset_index(drop=True), check_dtype=False)
    measured = metrics(scored)
    summary = read_json(directory/"summary.json")
    require(summary["forecast_metrics"] == measured and summary["simulation_metrics"] == state_metrics, "SUMMARY_ARITHMETIC")
    faults = read_json(directory/"faults.json")
    require(len(faults) >= 5 and all(x["accepted"] is True and x["label"] == LABELS["faults"] for x in faults), "FAULT_DISPOSITIONS")
    evidence = read_json(directory/"evidence.json")
    require(evidence["claim_labels"] == LABELS and all(r["outcome_label"] in LABELS.values() for r in evidence["records"]), "EVIDENCE_LABELS")
    # Optional current-state cross-check: independent warehouse snapshot + native inference.
    if data is not None:
        from demandos.serving.inputs import verify_sources
        from demandos.serving.registry import Registry
        from demandos.serving.prediction import predict
        from demandos.serving.features import validate_frame
        from .evidence import collect
        verify_sources(data)
        require(data.source_hashes == manifest["sources"] and data.model_hashes == manifest["models"], "INPUT_DIGEST_MISMATCH")
        require(code_identity(data.root)["files"] == manifest["code"]["files"], "CAPSTONE_CODE_CHANGED")
        require(prerequisites(data.root)["packages"] == manifest["packages"], "CAPSTONE_ENVIRONMENT_CHANGED")
        for relative, expected_hash in manifest["code"]["files"].items():
            require(file_hash(data.root/relative) == expected_hash, "CAPSTONE_CODE_CHANGED", relative)
        for relative, expected_hash in manifest["prior_registry"].items():
            require(file_hash(data.root/relative) == expected_hash, "HISTORICAL_REGISTRY_CHANGED", relative)
        for r in evidence["records"]:
            for relative, expected_hash in r["sources"].items():
                require(file_hash(data.root/relative) == expected_hash, "EVIDENCE_SOURCE_CHANGED", relative)
        current = data.snapshot.merge(frames["metadata"][KEY], on=KEY, validate="many_to_one")
        pd.testing.assert_frame_equal(current.reset_index(drop=True), source.reset_index(drop=True))
        features, frozen_digest = extract(current, data.calendar, frames["metadata"], CONFIG["origin"], data.feature_config, data.mapping)
        validate_frame(frames["features"], data.feature_config, data.mapping)
        parity(features, frames["features"])
        require(frozen_digest == manifest["eligible_snapshot_digest"], "CAPSTONE_PIT_SNAPSHOT")
        reg = Registry(data.root, directory/"registry", [data.root/"data/quantiles/global_h28",data.root/"data/conformal/m5_h28",data.root/"configs"])
        replay, _ = predict(reg, CONFIG["native_checkpoint"], features, current, manifest["feature_contract_digest"], approved=True)
        pd.testing.assert_frame_equal(replay, frames["forecasts"], check_dtype=False, check_exact=False, atol=1e-8, rtol=0)
        require(collect(data.root, digest(data.source_hashes)) == evidence, "EVIDENCE_ARITHMETIC_CHANGED")
    return dict(run_id=identifier, forecast_rows=len(scored), state_rows=len(main["daily"]), forecast_metrics=measured,
                simulation_metrics=state_metrics, independent_record_checks=True, current_sources_and_native_replay=data is not None,
                invariants=["forecast keys/dates/quantiles/provenance", "inventory/backlog/queue/cost conservation", "persistent shared cash/supplier/flow/store quotas",
                            "conservative physical storage ceilings", "unique simulated order IDs", "day-14 perturbation complete prefix", "mature labels", "scientific labels"])
