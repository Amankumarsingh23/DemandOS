"""Nested historical fitting, honest calibration ledgers and real M5 experiments."""

from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import numpy as np
import pandas as pd

from demandos.warehouse.cli import settings, build_artifacts, connection
from demandos.features.pipeline import totals
from demandos.features.validation import generate, supervised_labels
from demandos.boosting.dataset import chronological_development
from demandos.quantiles.data import prepare
from demandos.quantiles.pipeline import load_config as quantile_config, fit_trees, fit_neural
from demandos.quantiles.contracts import CASE, Q, repair, forecast_rows
from demandos.quantiles import neural
from demandos.quantiles.validation import checks as predictor_checks
from demandos.nbeats.dataset import TargetDataset
from demandos.neural.sequences import examples
from demandos.backtesting.metrics import horizon_band
from demandos.backtesting.pipeline import publish
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.reconciliation.pipeline import peak_memory
from .ledger import KEYS, scored
from .engine import static_intervals, aci_intervals
from .metrics import summarize, trajectory_coverage


def load_config(path):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"conformal"}:
        raise ValueError("Expected [conformal]")
    c = raw["conformal"]
    expected = {"alpha", "validation_origin", "fit_cutoff", "split_origins", "split_end", "issue_stride_days",
                "rolling_scores", "half_life_days", "aci_start", "aci_gamma", "aci_alpha_min", "aci_alpha_max",
                "path_min_origins", "path_min_trajectories", "seed"}
    if set(c) != expected or c["seed"] != 42 or not 0 < c["alpha"] < 1:
        raise ValueError("Missing/unknown/faulty fixed conformal configuration")
    for name, low, high in [("issue_stride_days", 7, 28), ("rolling_scores", 32, 2048),
                            ("half_life_days", 1, 365), ("path_min_origins", 2, 12), ("path_min_trajectories", 5, 100)]:
        if type(c[name]) is not int or not low <= c[name] <= high:
            raise ValueError(f"Invalid bounded {name}")
    if not 0 < c["aci_gamma"] <= .01 or not 0 < c["aci_alpha_min"] < c["alpha"] < c["aci_alpha_max"] < 1:
        raise ValueError("Invalid predeclared ACI bounds/gamma")
    dates = pd.DatetimeIndex(pd.to_datetime(c["split_origins"]))
    cutoff, selection_end, end = pd.Timestamp(c["fit_cutoff"]), pd.Timestamp(c["validation_origin"])+pd.Timedelta(days=28), pd.Timestamp(c["split_end"])
    if (dates.empty or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize()) or selection_end > cutoff or dates.min() < cutoff
            or np.any(np.diff(dates.as_unit("ns").asi8) < pd.Timedelta(days=28).value)
            or dates.max()+pd.Timedelta(days=28) > end or pd.Timestamp(c["aci_start"]) < cutoff):
        raise ValueError("Nested selection/calibration timing or nonoverlapping split blocks invalid")
    return c


def build_ledger(con, data, trees, model, config):
    frames = []
    values = None
    for column, tree in zip(Q, trees):
        current = tree.predict(data["inference"])[CASE+["raw_prediction"]].rename(columns={"raw_prediction": column})
        values = current if values is None else values.merge(current, on=CASE, validate="one_to_one")
    values["family"], values["model_id"] = "lightgbm", "s17_lightgbm_frozen_"+config["fit_cutoff"]
    values["fit_target_max"] = pd.Timestamp(trees[0].metadata["training_target_max"])
    frames.append(values)
    rows = examples(data["evaluation_keys"], data["issue_origins"], data["neural_config"])
    predictions = neural.predict(model, TargetDataset(data["panel"], rows, data["neural_config"]), data["neural_config"])
    values = forecast_rows(rows, predictions, "joint_quantiles", model.fit_cutoff, model.fit_target_max)
    values["family"], values["model_id"] = "neural", "s17_neural_frozen_"+config["fit_cutoff"]
    values = values.drop(columns=["model", "fit_cutoff"])
    frames.append(values)
    ledger = pd.concat(frames, ignore_index=True)
    for name in Q:
        ledger["raw_"+name] = ledger[name]
    ledger[Q] = repair(ledger[Q].to_numpy(float))
    labels = supervised_labels(con, data["inference"], data["origins"][-1]+pd.Timedelta(days=28))
    ledger = ledger.merge(labels.rename(columns={"target_units_sold": "actual"}), on=CASE, validate="many_to_one")
    ledger = ledger.merge(data["selected"], on=["store_id", "item_id"], validate="many_to_one")
    ledger["model_fit_cutoff"] = pd.Timestamp(config["fit_cutoff"])
    ledger["selection_cutoff"] = pd.Timestamp(config["validation_origin"])+pd.Timedelta(days=28)
    ledger["observation_available_date"] = ledger.target_date
    ledger["issue_date"] = ledger.origin_date
    ledger["prediction"] = ledger.q50
    ledger["horizon_band"] = horizon_band(ledger.horizon)
    ledger = ledger.sort_values(["family", "origin_date", "store_id", "item_id", "horizon"]).reset_index(drop=True)
    ledger["score_id"] = np.arange(len(ledger))
    return scored(ledger)


def reference_intervals(root, sources):
    directory = root/"data/quantiles/global_h28"
    summary = json.loads((directory/"summary.json").read_text())
    if summary["status"] != "passed" or summary["source_fingerprints"] != sources:
        raise ValueError("Season 16 reference source provenance mismatch")
    frame = pd.read_parquet(directory/"tables/forecasts.parquet")
    frame["lower"], frame["upper"] = frame.q10, frame.q90
    frame["method"] = frame.model.map({"lightgbm_quantiles": "season16_final_lightgbm_reference",
        "joint_nbeats_quantiles": "season16_final_neural_reference", "historical_empirical": "historical_empirical"})
    if frame.method.isna().any() or len(frame) != 6300:
        raise ValueError("Unexpected Season 16 reference coverage")
    frame["family"] = frame.model
    frame["status"] = "uncalibrated_reference"
    frame["horizon_band"] = horizon_band(frame.horizon)
    return frame[[*KEYS, "cohort", "method", "family", "lower", "upper", "horizon_band", "status"]], {
        "summary": fingerprint(directory/"summary.json"), "forecasts": fingerprint(directory/"tables/forecasts.parquet")}


def evaluate(intervals, ledger, diagnostics, alpha):
    if ledger.groupby(KEYS).actual.nunique(dropna=False).ne(1).any():
        raise ValueError("Forecast families have inconsistent actual targets")
    labels = ledger[[*KEYS, "actual"]].drop_duplicates(KEYS)
    output = intervals.merge(labels, on=KEYS, validate="many_to_one")
    output = output.merge(diagnostics[["store_id", "item_id", "origin_date", "regime"]],
                          on=["store_id", "item_id", "origin_date"], validate="many_to_one")
    if output.actual.isna().any() or output.duplicated([*KEYS, "method"]).any():
        raise ValueError("Invalid common forecast-observation keys")
    planned = {"continuity": 1092, "sparse_stress": 1008}
    wanted = ledger[ledger.origin_date.isin(diagnostics.origin_date.unique())][["cohort", *KEYS]].drop_duplicates()
    for (cohort, method), group in output.groupby(["cohort", "method"]):
        expected_keys = set(wanted[wanted.cohort.eq(cohort)][KEYS].itertuples(index=False, name=None))
        actual_keys = set(group[KEYS].itertuples(index=False, name=None))
        if len(group) != planned[cohort] or expected_keys != actual_keys:
            raise ValueError(f"Incomplete method coverage {cohort}/{method}")
    tables = {"intervals": output}
    for name, extra in {"metrics_overall": [], "metrics_origin": ["origin_date"], "metrics_horizon": ["horizon"],
                        "metrics_band": ["horizon_band"], "metrics_regime": ["regime"]}.items():
        tables[name] = summarize(output, ["cohort", "method", *extra], alpha)
    trajectories, joint = trajectory_coverage(output)
    tables.update(trajectory_cases=trajectories, metrics_trajectory=joint)
    # Pooled daily metrics must equal count-weighted horizon results.
    h = tables["metrics_horizon"]
    for column in ["coverage", "width_mean", "interval_score", "lower_miss_rate", "upper_miss_rate"]:
        weighted = h.assign(weight=h[column]*h.observations).groupby(["cohort", "method"]).agg(total=("weight", "sum"), n=("observations", "sum"))
        base = tables["metrics_overall"].set_index(["cohort", "method"]).reindex(weighted.index)
        np.testing.assert_allclose(weighted.total/weighted.n, base[column], rtol=1e-12, atol=1e-12)
    return tables


def calibrator_checks(data, ledger, config):
    gates = []
    baseline_static, baseline_thresholds, _ = static_intervals(ledger, data["origins"], config)
    for asof in data["origins"]:
        baseline_aci, _, baseline_events = aci_intervals(ledger, asof, config)
        for mode in ["mutate_future_sales", "remove_future_sales", "mutate_future_prices"]:
            daily = data["daily"].copy()
            future = daily.date.gt(asof)
            if mode == "mutate_future_sales": daily.loc[future, "units_sold"] += 1000000
            elif mode == "remove_future_sales": daily = daily.loc[~future].copy()
            else: daily.loc[future, "sell_price"] = 99999.
            labels = daily[["store_id", "item_id", "date", "units_sold"]].rename(columns={"date": "target_date", "units_sold": "actual"})
            altered = ledger.drop(columns=["actual", "cqr_signed", "cqr_nonnegative", "absolute"]).merge(labels,
                on=["store_id", "item_id", "target_date"], how="left", validate="many_to_one")
            new_static, new_thresholds, _ = static_intervals(altered, [asof], config)
            for first, second in [(baseline_static[baseline_static.origin_date.eq(asof)], new_static),
                                  (baseline_thresholds[baseline_thresholds.origin_date.eq(asof)], new_thresholds)]:
                pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
            new_aci, _, events = aci_intervals(altered, asof, config)
            pd.testing.assert_frame_equal(baseline_aci.reset_index(drop=True), new_aci.reset_index(drop=True))
            pd.testing.assert_frame_equal(baseline_events.reset_index(drop=True), events.reset_index(drop=True))
            gates.append({"origin_date": asof, "perturbation": mode, "unchanged_static_intervals_thresholds": True,
                          "unchanged_aci_intervals_and_state": True, "mature_feedback_events": len(events),
                          "future_score_candidates_excluded": True})
    return pd.DataFrame(gates)


def run(root, config):
    start = perf_counter()
    root, source, warehouse, env = settings(root)
    build_artifacts(root, source, warehouse)
    sources = [source, root/"data/interim/sales_long.parquet", *sorted((root/"data/raw").glob("*.csv"))]
    before = {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}
    target = root/"data/conformal/m5_h28"
    with connection(warehouse, env) as con, TemporaryDirectory(prefix=".conformal-", dir=root/"data") as temporary:
        temporary = Path(temporary)
        initial = totals(con)
        expected = json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(value) != str(expected[name]) for name, value in initial.items()):
            raise ValueError("Verified M5 warehouse baseline changed")
        print("Preparing bounded historical feature and sequence data", flush=True)
        qconfig = quantile_config(root/"configs/quantiles.toml")
        data = prepare(con, root, qconfig)
        data["quantile_config"] = qconfig
        reference, provenance = reference_intervals(root, before)
        data["boost"] = replace(data["boost"], validation_origin=config["validation_origin"])
        data["development_train"], data["development_validation"], data["validation_end"] = chronological_development(data["pool"], data["boost"], data["origins"][0])
        data["issue_origins"] = list(pd.date_range(config["fit_cutoff"], data["origins"][-1], freq=f"{config['issue_stride_days']}D"))
        if not set(data["origins"]).issubset(data["issue_origins"]) or pd.Timestamp(config["split_end"]) > data["origins"][0]:
            raise ValueError("Calibration schedule/evaluation origin mismatch")
        con.register("feature_keys", data["evaluation_keys"])
        data["inference"] = generate(con, data["issue_origins"], data["feature_config"], data["mapping"]).features
        cutoff = pd.Timestamp(config["fit_cutoff"])
        trees, tree_fits, selected = fit_trees(data, temporary, cutoff)
        model, neural_fits = fit_neural(data, temporary, cutoff)
        ledger = build_ledger(con, data, trees, model, config)
        print(f"Genuine held-out ledger: {len(ledger):,} forecasts; same frozen models throughout", flush=True)
        static, thresholds, availability = static_intervals(ledger, data["origins"], config)
        aci_all, aci_thresholds, events = aci_intervals(ledger, data["origins"][-1], config)
        aci = aci_all[aci_all.origin_date.isin(data["origins"])].copy()
        combined = pd.concat([static, aci, reference], ignore_index=True)
        tables = evaluate(combined, ledger, data["origin_diagnostics"], config["alpha"])
        tables.update(forecast_score_ledger=ledger, thresholds=pd.concat([thresholds, aci_thresholds], ignore_index=True),
                      aci_issued_intervals=aci_all, aci_feedback=events, path_availability=availability,
                      evaluation_cohort=data["selected"], training_cohort=data["training_cohort"], inference_features=data["inference"])
        predictor_gates = []
        for origin in data["origins"]:
            predictor_gates.extend(predictor_checks(con, data, origin, trees, model, data["neural_config"], qconfig["baseline_days"], cutoff))
            print(f"Frozen historical model future-input gates passed: {origin.date()}", flush=True)
        tables["predictor_checks"] = pd.DataFrame(predictor_gates)
        tables["calibrator_checks"] = calibrator_checks(data, ledger, config)
        print("All calibration maturity and future-perturbation gates passed", flush=True)
        if before != {p.relative_to(root).as_posix(): fingerprint(p) for p in sources} or totals(con) != initial:
            raise ValueError("Source fingerprints or warehouse totals changed")
        for name, path in [("summary", "summary.json"), ("forecasts", "tables/forecasts.parquet")]:
            if fingerprint(root/"data/quantiles/global_h28"/path) != provenance[name]:
                raise ValueError("Season 16 reference artifact changed")
        (temporary/"tables").mkdir()
        for name, frame in tables.items():
            frame.to_parquet(temporary/"tables"/f"{name}.parquet", index=False)
        summary = {"status": "passed", "config": config, "models": {"tree_fits": tree_fits, "neural_fits": neural_fits},
            "selected_rounds": selected, "selected_epochs": neural_fits[0]["selected_epochs"],
            "training_series": len(data["training_cohort"]), "model_selection_end": data["validation_end"],
            "historical_issue_origins": data["issue_origins"], "evaluation_origins": data["origins"],
            "methods": sorted(tables["intervals"].method.unique()), "metrics": tables["metrics_overall"].to_dict("records"),
            "trajectory_metrics": tables["metrics_trajectory"].to_dict("records"), "reference_provenance": provenance,
            "source_fingerprints": before, "warehouse_totals": initial, "inputs_unchanged": True,
            "table_rows": {name: len(frame) for name, frame in tables.items()},
            "runtime_seconds": perf_counter()-start, "peak_process_mib": peak_memory(),
            "warnings": sum(len(f["warnings"])+len(f["native_warning_lines"]) for f in tree_fits)+sum(len(f["warnings"]) for f in neural_fits),
            "failed_or_skipped_cases": 0, "path_availability": availability.to_dict("records"),
            "versions": {name: version(name) for name in ["numpy", "pandas", "lightgbm", "torch", "duckdb"]},
            "limitations": ["M5 temporal, cross-series and overlapping-horizon dependence does not establish exchangeability; no exact 80% conditional guarantee is claimed.",
                "Primary calibration and evaluation share new October-11-frozen models. Season 16 January-31-fitted outputs are separate unchanged references, not retrospectively calibrated with in-sample errors.",
                "The inherited continuity/stress universes are previously examined retrospective benchmarks; selection and repeated analysis limit prospective claims.",
                "Rolling 512-score windows, fixed 28-day recency half-life and bounded event-wise ACI are heuristic time-series adaptations. ACI bounds and delayed overlapping feedback differ from unbounded theoretical ACI.",
                "Scores are in original sales units, pooled within predeclared cohort/horizon bands. Different series scales and repeated target dates create heterogeneity/correlation.",
                "Pathwise coverage targets the full 28-day trajectory with a separate max-score threshold; pointwise coverage alone has no simultaneous interpretation.",
                "Known support clipping intersects with nonnegative observed sales; target-free midpoint enlargement handles signed-CQR empty sets conservatively.",
                "Forecasts describe recorded observed sales, not latent demand or verified stockouts. No inventory policies are implemented."]}
        from .report import artifacts
        artifacts(summary, tables, temporary)
        (temporary/"summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (temporary/"verification.json").write_text(json.dumps({"status": "passed", "predictor_future_gates": len(predictor_gates),
            "calibrator_future_gates": len(tables["calibrator_checks"]), "sources_unchanged": True,
            "warehouse_rows": initial["daily_rows"], "same_model_heldout_ledger": True, "mature_feedback_only": True}, indent=2)+"\n", encoding="utf-8")
        publish(temporary, target)
    return summary, target
