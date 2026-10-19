"""Real M5 training, exact-grain comparisons, and mandatory future-invariance gates."""

from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from demandos.backtesting.pipeline import publish
from demandos.boosting.config import load_config as load_boosting
from demandos.boosting.dataset import build, eligible_rows
from demandos.boosting.evaluation import statistical_predictions, attach_targets, comparison_tables
from demandos.boosting.models import GlobalModel, TARGET
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.features.pipeline import load_keys, totals
from demandos.warehouse.cli import settings, connection, build_artifacts
from .sequences import Panel, Preprocessor, SequenceDataset, examples, PAST, FUTURE
from .models import train, predict, save, load


MODELS = ("lstm", "gru", "lightgbm", "tsb", "seasonal_naive")
ATOL, RTOL = 1e-6, 1e-6


def prediction_rows(rows, predictions, architecture, horizon=28):
    if predictions.shape != (len(rows), horizon) or rows.duplicated(["store_id", "item_id", "origin_date"]).any():
        raise ValueError("Forecast matrix/key alignment failure")
    records = []
    for row, values in zip(rows.itertuples(), predictions):
        for h, value in enumerate(values, 1):
            records.append({"store_id": row.store_id, "item_id": row.item_id, "origin_date": row.origin_date,
                            "target_date": row.origin_date+pd.Timedelta(days=h), "horizon": h,
                            "model": architecture, "raw_prediction": float(value), "prediction": float(value)})
    return pd.DataFrame(records)


def assert_same_tensors(first, second):
    pd.testing.assert_frame_equal(first.rows, second.rows)
    if len(first.tensors) != len(second.tensors):
        raise ValueError("Dataset label contract changed")
    for a, b in zip(first.tensors, second.tensors):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def future_panels(daily, calendar, metadata, origin):
    changed = daily.copy()
    future = pd.to_datetime(changed.date).gt(origin)
    changed.loc[future, "units_sold"] += 1000000
    changed.loc[future, "sell_price"] = 99999.
    changed["target_units_sold"] = 999999.  # A deliberately forbidden extra column, never selected.
    return (Panel(changed, calendar, metadata),
            Panel(changed.loc[~future].copy(), calendar, metadata))


def history_diagnostics(panel, rows, processor, config):
    """Capability diagnostics only: no claim of held-out cold-start forecast quality."""
    first = rows.iloc[0]
    records = []
    origin = pd.Timestamp(first.origin_date)
    key = (first.store_id, first.item_id)
    dates, values = panel.series[key]
    keep = panel.metadata.loc[[key]].reset_index(drop=True)
    # Explicitly restrict visible support for this capability diagnostic only.
    keep = keep.drop(columns=["first_observed_date"], errors="ignore")
    calendar = panel.calendar
    for observed in (0, 3, 7, 28, 56):
        chosen = (dates <= origin) & (dates > origin-pd.Timedelta(days=observed))
        frame = pd.DataFrame({"store_id": first.store_id, "item_id": first.item_id,
                              "date": dates[chosen], "units_sold": values[chosen]})
        candidate = Panel(frame, calendar, keep)
        dataset = SequenceDataset(candidate, rows.iloc[:1], processor, config)
        audit = dataset.audits[0]
        if audit["history_observed"] != observed or not torch.isfinite(dataset.tensors[0]).all():
            raise ValueError("Short-history mask diagnostic failed")
        records.append({"observed_days": observed, **audit, "diagnostic_only": True})
    return pd.DataFrame(records)


def run(root, config):
    started = perf_counter()
    root, source, warehouse, env = settings(root)
    build_artifacts(root, source, warehouse)
    sources = [source, root/"data/interim/sales_long.parquet", *sorted((root/"data/raw").glob("*.csv"))]
    before = {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}
    boost = load_boosting(root/"configs/boosting.toml")
    target = root/"data/neural/global_h28"
    with connection(warehouse, env) as con, TemporaryDirectory(prefix=".neural-", dir=root/"data") as temp:
        temp = Path(temp)
        initial = totals(con)
        expected = json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(value) != str(expected[name]) for name, value in initial.items()):
            raise ValueError("Warehouse differs from verified source baseline")
        data = build(con, root, boost)
        # Reuse Season 9 feature generation once for the pooled baseline; neural inputs
        # are the bounded daily projection below, never tabular labels or EDA regimes.
        train_keys = load_keys(con, boost.selection_cutoff, data["training_cohort"])
        metadata = pd.concat([train_keys, data["evaluation_keys"]]).drop_duplicates(["store_id", "item_id"])
        con.register("neural_keys", metadata[["store_id", "item_id"]])
        daily = con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price
            from main.fact_daily_sales f join neural_keys k using(store_id,item_id)
            where date between ? and ? order by store_id,item_id,date""",
            [(data["training_origins"][0]-pd.Timedelta(days=config.lookback-1)).date(),
             (data["origins"][-1]+pd.Timedelta(days=config.horizon)).date()]).fetchdf()
        calendar = con.execute("select * from main.dim_calendar order by date").fetchdf()
        panel = Panel(daily, calendar, metadata)
        development_cutoff = pd.Timestamp(boost.validation_origin)
        validation_end = development_cutoff+pd.Timedelta(days=config.horizon)
        if validation_end >= data["origins"][0]:
            raise ValueError("Development validation overlaps retrospective evaluation")
        train_rows = examples(train_keys, data["training_origins"], config, development_cutoff)
        val_rows = examples(train_keys, [development_cutoff], config)
        processor = Preprocessor.fit(panel, train_rows, config, development_cutoff)
        development = SequenceDataset(panel, train_rows, processor, config, label_cutoff=development_cutoff)
        validation = SequenceDataset(panel, val_rows, processor, config, label_cutoff=validation_end)
        tables, selected, records, predictions, coverage, checks, diagnostics = {}, {}, [], [], [], [], []
        curves, tensor_audits, timings = [], [], []
        print(f"Neural cohort: {len(train_keys)} series; development {len(development)} windows, validation {len(validation)}; daily cache {len(daily):,}", flush=True)
        for architecture in ("lstm", "gru"):
            model, meta = train(architecture, processor, development, config, cutoff=development_cutoff, validation=validation)
            selected[architecture] = meta["selected_epochs"]
            path = temp/"models"/architecture/"development"
            save(model, processor, meta, path)
            reloaded, _, _ = load(path, device=config.device)
            # Probe inference without validation labels.
            probe = SequenceDataset(panel, val_rows, processor, config)
            np.testing.assert_allclose(predict(model, probe, config), predict(reloaded, probe, config), rtol=RTOL, atol=ATOL)
            records.append({**{k: v for k, v in meta.items() if k not in ("curves", "config")}, "phase": "development",
                            "model": architecture, "save_load_equal": True, "model_bytes": (path/"weights.pt").stat().st_size})
            curves.extend({**row, "model": architecture, "phase": "development", "cutoff": development_cutoff} for row in meta["curves"])
        # Freeze the baseline's rounds using the same existing December policy.
        baseline = GlobalModel("lightgbm", data["feature_columns"], boost, data["feature_config"])
        baseline.fit(data["development_train"], allowed_cutoff=development_cutoff,
                     validation=data["development_validation"], validation_cutoff=validation_end)
        baseline_rounds = baseline.rounds
        baseline.save(temp/"models/lightgbm/development")
        print(f"Frozen neural epochs {selected}; LightGBM rounds {baseline_rounds}; no evaluation-target selection", flush=True)
        for origin in data["origins"]:
            rows = examples(train_keys, data["training_origins"], config, origin)
            inference_rows = examples(data["evaluation_keys"], [origin], config)
            prep = Preprocessor.fit(panel, rows, config, origin)
            fitting = SequenceDataset(panel, rows, prep, config, label_cutoff=origin)
            inference = SequenceDataset(panel, inference_rows, prep, config)
            modified, removed = future_panels(daily, calendar, metadata, origin)
            altered = SequenceDataset(modified, inference_rows, prep, config)
            deleted = SequenceDataset(removed, inference_rows, prep, config)
            assert_same_tensors(inference, altered); assert_same_tensors(inference, deleted)
            # Even regenerating all eligible training labels is invariant to future changes.
            assert_same_tensors(fitting, SequenceDataset(modified, rows, prep, config, label_cutoff=origin))
            assert_same_tensors(fitting, SequenceDataset(removed, rows, prep, config, label_cutoff=origin))
            diagnostics.append(history_diagnostics(panel, inference_rows, prep, config).assign(origin=origin))
            tensor_audits.extend({**audit, "fit_cutoff": origin, "role": "inference"} for audit in inference.audits)
            for architecture in ("lstm", "gru"):
                model, meta = train(architecture, prep, fitting, config, cutoff=origin, epochs=selected[architecture])
                path = temp/"models"/architecture/f"origin_{origin:%Y%m%d}"
                save(model, prep, meta, path)
                start = perf_counter()
                values = predict(model, inference, config)
                inference_seconds = perf_counter()-start
                loaded, saved_prep, _ = load(path, device=config.device)
                saved_input = SequenceDataset(panel, inference_rows, saved_prep, config)
                np.testing.assert_allclose(values, predict(loaded, saved_input, config), rtol=RTOL, atol=ATOL)
                differences = []
                for change in (altered, deleted):
                    other = predict(loaded, change, config)
                    np.testing.assert_allclose(values, other, rtol=RTOL, atol=ATOL)
                    differences.append(float(np.abs(values-other).max()))
                checks.append({"model": architecture, "origin": origin, "windows": len(inference), "rows": values.size,
                               "tensors_exact": True, "future_sales_mutated_and_removed": True, "future_prices_mutated": True,
                               "future_labels_ignored": True, "eligible_training_unchanged": True,
                               "save_load_equal": True, "max_prediction_difference": max(differences), "rtol": RTOL, "atol": ATOL})
                predictions.append(prediction_rows(inference.rows, values, architecture))
                for row in data["evaluation_cohort"].itertuples():
                    coverage.append({"fit_id": f"{architecture}/{origin:%Y%m%d}", "model": architecture,
                                     "origin": origin, "series_id": row.series_id, "status": "ok", "reason": None})
                records.append({**{k: v for k, v in meta.items() if k not in ("curves", "config")}, "phase": "evaluation_refit",
                                "model": architecture, "save_load_equal": True, "model_bytes": (path/"weights.pt").stat().st_size})
                curves.extend({**row, "model": architecture, "phase": "evaluation_refit", "cutoff": origin} for row in meta["curves"])
                timings.append({"model": architecture, "origin": origin, "fit_seconds": meta["fit_seconds"],
                                "inference_seconds": inference_seconds, "windows": len(fitting), "forecast_rows": values.size})
            tabular = eligible_rows(data["pool"], origin)
            features = data["inference"].loc[data["inference"].origin_date.eq(origin)]
            baseline = GlobalModel("lightgbm", data["feature_columns"], replace(boost, iterations=baseline_rounds), data["feature_config"])
            baseline.fit(tabular, allowed_cutoff=origin)
            baseline.save(temp/"models/lightgbm"/f"origin_{origin:%Y%m%d}")
            start = perf_counter()
            predictions.append(baseline.predict(features))
            timings.append({"model": "lightgbm", "origin": origin, "fit_seconds": baseline.metadata["fit_seconds"],
                            "inference_seconds": perf_counter()-start, "windows": None, "forecast_rows": len(features)})
            for row in data["evaluation_cohort"].itertuples():
                coverage.append({"fit_id": f"lightgbm/{origin:%Y%m%d}", "model": "lightgbm", "origin": origin,
                                 "series_id": row.series_id, "status": "ok", "reason": None})
            print(f"Verified {origin:%Y-%m-%d}: {len(fitting):,} full training vectors; both models future-invariant", flush=True)
        stats, statistical_fits, statistical_coverage = statistical_predictions(data)
        predictions.append(stats[stats.model.isin(MODELS)])
        coverage.extend(row for row in statistical_coverage if row["model"] in MODELS)
        forecasts, scales = attach_targets(pd.concat(predictions, ignore_index=True), coverage, data, MODELS)
        tables.update(comparison_tables(forecasts, MODELS))
        if len(tables["comparison_membership"]) != 1092 or len(forecasts) != len(MODELS)*1092:
            raise ValueError("Continuity benchmark keys changed or comparison coverage incomplete")
        tables.update(forecasts=forecasts, scales=scales, fit_status=pd.DataFrame(records),
                      baseline_fit_status=pd.DataFrame([r for r in statistical_fits if r["model"] in MODELS]),
                      training_curves=pd.DataFrame(curves), tensor_audits=pd.DataFrame(tensor_audits),
                      leakage_checks=pd.DataFrame(checks), history_diagnostics=pd.concat(diagnostics, ignore_index=True),
                      timings=pd.DataFrame(timings), training_cohort=data["training_cohort"],
                      evaluation_cohort=data["evaluation_cohort"])
        eligibility = []
        for cutoff in [development_cutoff, *data["origins"]]:
            included = examples(train_keys, data["training_origins"], config, cutoff)
            eligibility.append({"fit_cutoff": cutoff, "candidate_windows": len(train_keys)*len(data["training_origins"]),
                                "eligible_windows": len(included), "label_values": len(included)*config.horizon,
                                "excluded_incomplete_vectors": len(train_keys)*len(data["training_origins"])-len(included),
                                "exclusion_reason": "target_end_after_fit_cutoff",
                                "history_start_min": included.origin_date.min()-pd.Timedelta(days=config.lookback-1),
                                "history_end_max": included.origin_date.max(),
                                "target_start_min": included.origin_date.min()+pd.Timedelta(days=1),
                                "target_end_max": included.origin_date.max()+pd.Timedelta(days=config.horizon)})
        tables["eligibility"] = pd.DataFrame(eligibility)
        if totals(con) != initial or before != {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}:
            raise ValueError("Input fingerprints or warehouse totals changed")
        (temp/"tables").mkdir()
        # Nested per-model metadata belongs in JSON; retain scalar fit summaries in Parquet.
        fit_frame = tables["fit_status"]
        for column in fit_frame:
            if fit_frame[column].map(lambda value: isinstance(value, (list, dict))).any():
                fit_frame[column] = fit_frame[column].map(lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value)
        for name, frame in tables.items():
            frame.to_parquet(temp/"tables"/f"{name}.parquet", index=False)
        summary = {"status": "passed", "config": asdict(config), "boosting_config": asdict(boost),
                   "training_series": len(train_keys), "daily_cache_rows": len(daily), "union_series": len(metadata),
                   "development_windows": len(development), "development_labels": len(development)*config.horizon,
                   "validation_windows": len(validation), "development_fit_cutoff": development_cutoff,
                   "eligibility": eligibility,
                   "validation_end": validation_end, "evaluation_origins": data["origins"],
                   "selected_epochs": selected, "lightgbm_rounds": baseline_rounds,
                   "full_vector_eligibility": "origin + 28 days <= fit cutoff; incomplete vectors excluded",
                   "metrics_common": tables["metrics_common"].to_dict("records"),
                   "table_rows": {name: len(frame) for name, frame in tables.items()},
                   "tensor_shapes": {"past": ["B", config.lookback, len(PAST)], "future": ["B", config.horizon, len(FUTURE)],
                                     "static": ["B", len(STATIC)], "target_prediction": ["B", config.horizon]},
                   "feature_policy": "History-only observed sales/scale/mask; prices unavailable with mask; weekday/month/events/state SNAP; future calendar schedule assumed known as Season 8",
                   "normalization": "Global mean/std fit on eligible window history only; per-window mean floor 1 from own origin history; labels divided by same scale; outputs inverse-scaled; distinct from RMSSE denominator",
                   "source_fingerprints": before, "inputs_unchanged": True, "warehouse_totals": initial,
                   "invariance": checks, "fit_records": records, "runtime_seconds": perf_counter()-started,
                   "versions": {name: version(name) for name in ("torch", "numpy", "pandas", "duckdb", "lightgbm")},
                   "resources": {"python": platform.python_version(), "device": config.device, "torch_threads": config.threads,
                                 "workers": 0, "mixed_precision": False, "tensor_cache_bytes_max": max(sum(t.numel()*t.element_size() for t in fitting.tensors), sum(t.numel()*t.element_size() for t in development.tensors)),
                                 "peak_memory": "Not sampled; tensor-cache sizes and bounded warehouse settings reported instead"},
                   "warnings": ["Retrospective, previously examined small benchmark; no prospective or full-M5 ranking",
                                "Observed sales, not verified unconstrained demand; no stockout/inventory-savings claim",
                                "Cold/short-history masks verified; diagnostics do not validate cold-start forecast accuracy",
                                "CPU deterministic settings; cross-version/device bitwise reproducibility not promised"]}
        from .report import artifacts
        artifacts(summary, tables, temp)
        (temp/"summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        publish(temp, target)
    return summary, target
