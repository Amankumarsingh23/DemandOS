"""Bounded real training, frozen evaluation, hard safety gates and artifact publication."""

from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import numpy as np
import pandas as pd

from demandos.warehouse.cli import settings, connection, build_artifacts
from demandos.features.pipeline import totals
from demandos.features.validation import supervised_labels
from demandos.boosting.dataset import eligible_rows, row_digest
from demandos.boosting.models import TARGET
from demandos.nbeats.dataset import TargetDataset
from demandos.neural.sequences import examples
from demandos.backtesting.pipeline import publish
from demandos.backtesting.metrics import horizon_band
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.reconciliation.pipeline import peak_memory
from .contracts import LEVELS, Q, CASE, levels, validate, matched, repair, forecast_rows
from .metrics import summarize
from .tree import QuantileLightGBM
from . import neural
from .baseline import historical
from .data import prepare, hierarchy_diagnostic
from .validation import checks

MODELS = ("lightgbm_quantiles", "joint_nbeats_quantiles", "historical_empirical")


def load_config(path):
    with Path(path).open("rb") as handle:
        document = tomllib.load(handle)
    if set(document) != {"quantiles"}:
        raise ValueError("Expected [quantiles] config")
    config = document["quantiles"]
    if set(config) != {"levels", "fit_cutoff", "baseline_days", "stress_series", "epochs", "patience", "seed"}:
        raise ValueError("Missing/unknown quantile setting")
    if tuple(levels(config["levels"])) != LEVELS or config["seed"] != 42:
        raise ValueError("This benchmark fixes levels .1/.5/.9 and seed 42")
    for name, maximum in [("baseline_days", 365), ("stress_series", 24), ("epochs", 15), ("patience", 5)]:
        if type(config[name]) is not int or not 1 <= config[name] <= maximum:
            raise ValueError(f"Invalid bounded {name}")
    cutoff = pd.Timestamp(config["fit_cutoff"])
    if pd.isna(cutoff) or cutoff != cutoff.normalize():
        raise ValueError("Explicit daily frozen fit cutoff required")
    return config


def fit_trees(data, output, cutoff):
    training = eligible_rows(data["pool"], cutoff)
    trees, records, selected = [], [], {}
    for alpha in LEVELS:
        name = f"p{round(alpha*100):02d}"
        model = QuantileLightGBM(alpha, data["feature_columns"], data["boost"], data["feature_config"])
        model.fit(data["development_train"], allowed_cutoff=data["boost"].validation_origin,
                  validation=data["development_validation"], validation_cutoff=data["validation_end"])
        model.save(output/"models/lightgbm"/name/"development")
        selected[name] = model.rounds
        records.append({"model": "lightgbm_quantiles", "quantile": alpha, "phase": "development", **model.metadata})
        refit = QuantileLightGBM(alpha, data["feature_columns"], replace(data["boost"], iterations=model.rounds), data["feature_config"])
        refit.fit(training, allowed_cutoff=cutoff)
        directory = output/"models/lightgbm"/name/"frozen_evaluation"
        refit.save(directory)
        restored = QuantileLightGBM.load(directory)
        np.testing.assert_array_equal(refit.predict(data["inference"]).raw_prediction, restored.predict(data["inference"]).raw_prediction)
        trees.append(restored)
        records.append({"model": "lightgbm_quantiles", "quantile": alpha, "phase": "frozen_evaluation", **refit.metadata})
        print(f"Real quantile LightGBM {name}: {model.rounds}/{data['boost'].iterations} rounds; {len(training):,} fitting labels", flush=True)
    return trees, records, selected


def fit_neural(data, output, cutoff):
    config = data["neural_config"]
    training_keys = data["training_cohort"][["store_id", "item_id"]]
    development_cutoff = pd.Timestamp(data["boost"].validation_origin)
    training_rows = examples(training_keys, data["training_origins"], config, cutoff=development_cutoff)
    development = TargetDataset(data["panel"], training_rows, config, label_cutoff=development_cutoff)
    validation_rows = examples(training_keys, [development_cutoff], config)
    validation = TargetDataset(data["panel"], validation_rows, config, label_cutoff=data["validation_end"])
    model, meta = neural.train(development, config, cutoff=development_cutoff, validation=validation, validation_cutoff=data["validation_end"])
    neural.save(model, meta, output/"models/neural/development")
    final_rows = examples(training_keys, data["training_origins"], config, cutoff=cutoff)
    training = TargetDataset(data["panel"], final_rows, config, label_cutoff=cutoff)
    model, final = neural.train(training, config, cutoff=cutoff, epochs=meta["selected_epochs"])
    directory = output/"models/neural/frozen_evaluation"
    neural.save(model, final, directory)
    restored, _ = neural.load(directory)
    probe = TargetDataset(data["panel"], examples(data["evaluation_keys"], data["origins"], config), config)
    np.testing.assert_array_equal(neural.predict(model, probe, config), neural.predict(restored, probe, config))
    return restored, [{"phase": "development", **meta}, {"phase": "frozen_evaluation", **final}]


def forecast(data, trees, model, fit_cutoff):
    inference = data["inference"]
    values = None
    for name, tree in zip(Q, trees):
        result = tree.predict(inference)[CASE+["raw_prediction"]].rename(columns={"raw_prediction": name})
        values = result if values is None else values.merge(result, on=CASE, validate="one_to_one")
    values["model"], values["fit_cutoff"], values["fit_target_max"] = MODELS[0], fit_cutoff, pd.Timestamp(trees[0].metadata["training_target_max"])
    rows = examples(data["evaluation_keys"], data["origins"], data["neural_config"])
    dataset = TargetDataset(data["panel"], rows, data["neural_config"])
    predictions = neural.predict(model, dataset, data["neural_config"])
    neural_values = forecast_rows(rows, predictions, MODELS[1], model.fit_cutoff, model.fit_target_max)
    baseline, provenance = [], []
    for row in rows.itertuples():
        q, audit = historical(data["histories"][(row.store_id, row.item_id)], row.origin_date, data["quantile_config"]["baseline_days"])
        single = pd.DataFrame([{"store_id": row.store_id, "item_id": row.item_id, "origin_date": row.origin_date}])
        baseline.append(forecast_rows(single, q[None], MODELS[2], audit["fit_cutoff"], audit["fit_target_max"]))
        provenance.append({"store_id": row.store_id, "item_id": row.item_id, "origin_date": row.origin_date, **audit})
    return pd.concat([values, neural_values, *baseline], ignore_index=True), pd.DataFrame(provenance)


def evaluation_tables(raw):
    validate(raw, operational=False)
    matched(raw, MODELS)
    final = raw.copy()
    final[Q] = repair(raw[Q].to_numpy(float))
    validate(final)
    views = pd.concat([raw.assign(view="raw"), final.assign(view="repaired")], ignore_index=True)
    views["horizon_band"] = horizon_band(views.horizon)
    views["history_zero_band"] = np.where(views.zero_sales_fraction.ge(.8), "history_zero_at_least_80pct", "history_zero_under_80pct")
    views["target_status"] = np.where(views.actual.eq(0), "zero_observed_sale", "positive_observed_sale")
    tables = {"raw_forecasts": raw, "forecasts": final}
    for name, extra in {"metrics_overall": [], "metrics_origin": ["origin_date"], "metrics_horizon": ["horizon"],
                        "metrics_band": ["horizon_band"], "metrics_regime": ["regime"],
                        "metrics_zero_history": ["history_zero_band"], "metrics_target_status": ["target_status"]}.items():
        tables[name] = summarize(views, ["cohort", "view", "model", *extra])
    # Pooled pointwise losses must equal the count-weighted horizon decomposition.
    keys = ["cohort", "view", "model"]
    h = tables["metrics_horizon"].copy()
    for column in ["pinball_q10", "pinball_q50", "pinball_q90", "pinball_mean", "coverage_q10", "coverage_q50", "coverage_q90"]:
        h["weighted"] = h[column]*h.observations
        totals = h.groupby(keys).agg(total=("weighted", "sum"), n=("observations", "sum"))
        pooled = tables["metrics_overall"].set_index(keys).reindex(totals.index)
        np.testing.assert_allclose(totals.total/totals.n, pooled[column], rtol=1e-12, atol=1e-12)
    return tables


def run(root, config):
    started = perf_counter()
    root, source, warehouse, env = settings(root)
    build_artifacts(root, source, warehouse)
    sources = [source, root/"data/interim/sales_long.parquet", *sorted((root/"data/raw").glob("*.csv"))]
    before = {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}
    target = root/"data/quantiles/global_h28"
    with connection(warehouse, env) as con, TemporaryDirectory(prefix=".quantiles-", dir=root/"data") as temporary:
        temporary = Path(temporary)
        initial = totals(con)
        expected = json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(v) != str(expected[k]) for k, v in initial.items()):
            raise ValueError("Verified M5 warehouse baseline changed")
        print("Preparing real origin-safe features and history windows", flush=True)
        data = prepare(con, root, config)
        data["quantile_config"] = config
        cutoff = pd.Timestamp(config["fit_cutoff"])
        if not data["validation_end"] < cutoff <= data["origins"][0]:
            raise ValueError("Frozen evaluation fit cutoff must follow validation and precede evaluation targets")
        hierarchy, hierarchy_provenance = hierarchy_diagnostic(root, data, before, config["baseline_days"])
        trees, tree_fits, selected = fit_trees(data, temporary, cutoff)
        model, neural_fits = fit_neural(data, temporary, cutoff)
        tick = perf_counter()
        raw, baseline = forecast(data, trees, model, cutoff)
        inference_seconds = perf_counter()-tick
        labels = supervised_labels(con, data["inference"], data["origins"][-1]+pd.Timedelta(days=28))
        raw = raw.merge(labels, on=CASE, validate="many_to_one").rename(columns={TARGET: "actual"})
        raw = raw.merge(data["origin_diagnostics"], on=["store_id", "item_id", "origin_date"], validate="many_to_one")
        expected_cases = (13+config["stress_series"])*3*28
        if raw.groupby("model").size().ne(expected_cases).any() or len(raw) != expected_cases*3:
            raise ValueError("Incomplete required quantile model/target coverage")
        tables = evaluation_tables(raw)
        tables.update(baseline_provenance=baseline, origin_diagnostics=data["origin_diagnostics"], hierarchy_diagnostic=hierarchy,
                      training_cohort=data["training_cohort"], evaluation_cohort=data["selected"],
                      inference_audit=data["inference"])
        gates = []
        for origin in data["origins"]:
            gates.extend(checks(con, data, origin, trees, model, data["neural_config"], config["baseline_days"], cutoff))
            print(f"Quantile future-sales deletion/mutation and future-price gates passed: {origin.date()}", flush=True)
        tables["leakage_checks"] = pd.DataFrame(gates)
        if before != {p.relative_to(root).as_posix(): fingerprint(p) for p in sources} or totals(con) != initial:
            raise ValueError("Source fingerprint/warehouse totals changed")
        if fingerprint(root/"data/reconciliation/m5_cohort_h28/summary.json") != hierarchy_provenance["hierarchy_summary"]:
            raise ValueError("Prior hierarchy artifact changed")
        (temporary/"tables").mkdir()
        for name, frame in tables.items():
            frame.to_parquet(temporary/"tables"/f"{name}.parquet", index=False)
        (temporary/"feature_mapping.json").write_text(json.dumps(data["mapping"].as_dict(), indent=2)+"\n", encoding="utf-8")
        summary = {"status": "passed", "config": config, "levels": list(LEVELS), "models": list(MODELS),
            "fit_policy": "December validation selects iterations/epochs; evaluation weights/vocabularies fit once through 2016-01-31 and remain frozen at all three origins. Later sales update inference histories only.",
            "training_series": len(data["training_cohort"]), "development_tree_rows": len(data["development_train"]),
            "validation_tree_rows": len(data["development_validation"]), "final_tree_rows": len(eligible_rows(data["pool"], cutoff)),
            "fit_label_digest": row_digest(eligible_rows(data["pool"], cutoff)), "tree_fits": tree_fits,
            "neural_fits": neural_fits, "selected_rounds": selected, "selected_epochs": neural_fits[0]["selected_epochs"],
            "validation_origin": data["boost"].validation_origin, "validation_end": data["validation_end"], "fit_cutoff": cutoff,
            "evaluation_origins": data["origins"], "continuity_series": 13, "stress_series": config["stress_series"],
            "stress_rule": "Lowest prefix positive-day fractions among the established 96 training series through 2014-01-04, excluding continuity IDs; ties sorted store/item.",
            "feature_columns": data["feature_columns"], "feature_count": len(data["feature_columns"]),
            "coverage": 1., "skips": 0, "failed_models": 0, "fitted_model_count": len(tree_fits)+len(neural_fits),
            "tree_warning_count": sum(len(f["warnings"])+len(f["native_warning_lines"]) for f in tree_fits),
            "neural_warning_count": sum(len(f["warnings"]) for f in neural_fits),
            "metrics": tables["metrics_overall"].to_dict("records"), "hierarchy": hierarchy_provenance,
            "source_fingerprints": before, "warehouse_totals": initial, "inputs_unchanged": True,
            "table_rows": {name: len(frame) for name, frame in tables.items()}, "leakage_checks": gates,
            "inference_seconds": inference_seconds, "runtime_seconds": perf_counter()-started, "peak_process_mib": peak_memory(),
            "versions": {name: version(name) for name in ["numpy", "pandas", "lightgbm", "torch", "duckdb", "scikit-learn"]},
            "limitations": ["Forecasts concern recorded observed sales; no latent-demand or verified stockout inference.",
                "Previously studied retrospective 13-series benchmark, not a pristine prospective evaluation or official probabilistic M5 score.",
                "Inclusive C(tau)=mean(Y<=Q_tau) can exceed nominal tau at discrete zero atoms/ties; it is not a claim of exact conditional calibration.",
                "Wilson intervals are illustrative iid-binomial summaries; serial/horizon/cross-series correlation invalidates a general independent-trials interpretation.",
                "The neural softplus head can approach zero but does not parameterize exact zero ties; sparse-series behavior must be examined separately.",
                "Raw crossed P10>P90 endpoints have invalid intervals: their interval scores/widths/coverage are excluded with explicit counts; raw pinball and marginal coverage still use every case.",
                "The historical baseline uses the past 56-day unconditional empirical distribution, including zeros; it has no horizon-specific dynamics.",
                "No calibration correction, conformal algorithm, joint scenario model, quantile reconciliation or inventory optimization is implemented."]}
        from .report import artifacts
        artifacts(summary, tables, temporary)
        (temporary/"summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (temporary/"verification.json").write_text(json.dumps({"status": "passed", "future_checks": gates,
            "sources_unchanged": True, "warehouse_rows": initial["daily_rows"], "matched_cases_per_model": expected_cases,
            "serialization_checks": 4, "pooled_horizon_metrics_reconcile": True}, default=str, indent=2)+"\n", encoding="utf-8")
        publish(temporary, target)
    return summary, target
