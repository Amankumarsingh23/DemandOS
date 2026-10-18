"""Bounded extraction, fresh fits per origin/window, auditable planned-case outputs."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.eda.queries import fetch_frame
from demandos.forecasting.models import forecast, at_origin, ForecastResult
from demandos.forecasting.pipeline import parameters, sarima_cohort
from demandos.timeseries.cohort import training_labels, select_cohort
from demandos.warehouse.cli import settings, connection, build_artifacts
from .design import schedules
from .metrics import rmsse_scale, summarize, common_cases, paired_losses, horizon_band, CASE


def extract(con, root, config):
    dates = fetch_frame(con, "select date from main.dim_calendar order by date").date
    schedule = schedules(dates, config)
    first = schedule.origin.min()
    labels, products, label_config = training_labels(con, root, pd.Timestamp(dates.min()), first)
    chosen = select_cohort(labels, config.max_store_items)
    con.register("backtest_keys", chosen[["store_id", "item_id"]])
    panel = fetch_frame(con, """select f.store_id, f.item_id, f.date, f.units_sold
        from main.fact_daily_sales f join backtest_keys k using(store_id,item_id)
        order by f.store_id, f.item_id, f.date""")
    frames, metadata = {}, []
    for row in chosen.itertuples():
        series_id = f"store_item/{row.store_id}/{row.item_id}"
        frames[series_id] = panel[panel.store_id.eq(row.store_id) & panel.item_id.eq(row.item_id)][["date", "units_sold"]].copy()
        metadata.append({"series_id": series_id, "kind": "store_item", "store_id": row.store_id, "item_id": row.item_id,
                         "regime": row.regime, "abc_xyz": row.abc_xyz, "selection_reason": row.selection_reason,
                         "segment_fit_end": first})
    for grain, key, limit in (("store", "store_id", config.store_aggregates), ("category", "cat_id", config.category_aggregates)):
        selected = con.execute(f"select {key} from eda_{grain}_daily group by {key} order by sum(units_sold) desc, {key} limit ?", [limit]).fetchall()
        for identifier, in selected:
            series_id = f"{grain}/{identifier}"
            frames[series_id] = fetch_frame(con, f"select date, units_sold from main.mart_{grain}_daily where {key}=? order by date", [identifier])
            metadata.append({"series_id": series_id, "kind": grain, "store_id": identifier if grain == "store" else None,
                             "item_id": None, "regime": None, "abc_xyz": None,
                             "selection_reason": "earliest_origin_training_volume", "segment_fit_end": first})
    for frame in frames.values():
        frame["date"] = pd.to_datetime(frame.date)
    return frames, pd.DataFrame(metadata), schedule, labels, products, asdict(label_config)


def execute(frames, cohort, schedule, config):
    """Precompute targets; fit existing models on independent training prefixes."""
    if cohort.series_id.duplicated().any() or set(cohort.series_id) != set(frames):
        raise ValueError("Cohort identities must match unique input series")
    if not cohort.segment_fit_end.le(schedule.origin.min()).all():
        raise ValueError("Cohort labels use future data")
    allowed = {"arima": sarima_cohort(cohort, config.arima_series), "sarima": sarima_cohort(cohort, config.sarima_series)}
    forecasts, statuses, scales, training = [], [], [], []
    # Construct and validate ALL targets before fitting any model.
    target_cache = {}
    for row in schedule.itertuples():
        targets = pd.date_range(row.target_start, periods=config.horizon)
        for series_id, frame in frames.items():
            if frame.date.duplicated().any():
                raise ValueError("Duplicate observed series dates")
            target = frame.set_index("date").reindex(targets).units_sold.to_numpy(float)
            if not np.isfinite(target).all() or (target < 0).any():
                raise ValueError(f"Missing or invalid planned targets: {series_id}/{row.origin}")
            target_cache[(series_id, row.origin)] = target.copy()
    for row in schedule.itertuples():
        for member in cohort.itertuples():
            history = at_origin(frames[member.series_id], row.origin,
                                config.sliding_days if row.window == "sliding" else None)
            if history.index[0] != row.history_start or len(history) != row.training_days:
                raise ValueError("Actual training bounds do not match schedule")
            scale = rmsse_scale(history.to_numpy(float))
            scales.append({"window": row.window, "series_id": member.series_id, "origin": row.origin,
                           "kind": member.kind, "fit_start": history.index[0], "fit_end": history.index[-1],
                           "training_days": len(history), **scale})
            training.append(pd.DataFrame({"window": row.window, "origin": row.origin, "series_id": member.series_id,
                                          "date": history.index, "units_sold": history.to_numpy()}))
            for model in config.models:
                params = parameters(model, config.model_config)
                result = (ForecastResult(model, "skipped", "outside_earliest_origin_model_cap")
                          if model in allowed and member.series_id not in allowed[model] else
                          forecast(history, config.horizon, model, seasonal_period=config.model_config.seasonal_period, params=params))
                status = {"window": row.window, "series_id": member.series_id, "kind": member.kind,
                          "regime": member.regime, "abc_xyz": member.abc_xyz, "origin": row.origin, "model": model,
                          "status": result.status, "reason": result.reason, "runtime_seconds": result.runtime_seconds,
                          "warning_count": len(result.warnings), "warnings": json.dumps(result.warnings),
                          "model_metadata": json.dumps(json_ready(result.metadata), default=str, allow_nan=False),
                          "configuration": json.dumps(params), "fit_start": history.index[0], "fit_end": history.index[-1],
                          "training_days": len(history), "clipped_count": result.metadata.get("clipped_count", 0)}
                statuses.append(status)
                actuals = target_cache[(member.series_id, row.origin)]
                for h in range(1, config.horizon + 1):
                    ok = result.status == "ok"
                    raw = float(result.raw_predictions[h - 1]) if ok else np.nan
                    prediction = float(result.predictions[h - 1]) if ok else np.nan
                    actual = actuals[h - 1]
                    forecasts.append({**{key: status[key] for key in ("window", "series_id", "kind", "regime", "abc_xyz", "origin", "model", "status", "reason", "runtime_seconds", "configuration")},
                                      "horizon": h, "target_date": row.origin + pd.Timedelta(days=h),
                                      "actual": actual, "raw_prediction": raw, "prediction": prediction,
                                      "absolute_error": abs(actual - prediction), "squared_error": (actual - prediction) ** 2,
                                      "scale": scale["scale"], "scale_status": scale["scale_status"]})
        print(f"Backtested {row.window} {row.origin:%Y-%m-%d}", flush=True)
    outputs = {"forecasts": pd.DataFrame(forecasts), "fit_status": pd.DataFrame(statuses),
               "scales": pd.DataFrame(scales), "training": pd.concat(training, ignore_index=True)}
    validate_outputs(outputs, cohort, schedule, config)
    return outputs, allowed


def validate_outputs(outputs, cohort, schedule, config):
    frame, status = outputs["forecasts"], outputs["fit_status"]
    expected = len(cohort) * len(schedule) * len(config.models)
    keys = ["window", "series_id", "origin", "model"]
    if len(status) != expected or status.duplicated(keys).any() or len(frame) != expected * config.horizon or frame.duplicated(CASE + ["model"]).any():
        raise ValueError("Planned forecast/status counts or uniqueness do not reconcile")
    counts = frame.groupby(keys).horizon.agg(["count", "min", "max"])
    if not (counts["count"].eq(config.horizon) & counts["min"].eq(1) & counts["max"].eq(config.horizon)).all():
        raise ValueError("Incomplete horizon vectors")
    if not frame.target_date.eq(frame.origin + pd.to_timedelta(frame.horizon, unit="D")).all() or not status.fit_end.eq(status.origin).all():
        raise ValueError("Forecast/fit boundary mismatch")
    if not set(frame.status).issubset({"ok", "failed", "skipped"}):
        raise ValueError("Unknown fit outcome")
    merged = frame.merge(status[keys + ["status"]], on=keys, validate="many_to_one", suffixes=("", "_fit"))
    if not merged.status.eq(merged.status_fit).all():
        raise ValueError("Fit and forecast outcomes disagree")
    ok = frame.status.eq("ok")
    if not np.isfinite(frame.loc[ok, ["actual", "prediction", "raw_prediction", "absolute_error", "squared_error"]].to_numpy(float)).all():
        raise ValueError("Invalid successful prediction values")
    np.testing.assert_array_equal(frame.loc[ok, "prediction"], np.maximum(frame.loc[ok, "raw_prediction"], 0))
    if frame.loc[~ok, ["prediction", "raw_prediction"]].notna().any().any():
        raise ValueError("Failed/skipped cases contain invented forecasts")
    if not np.isfinite(frame.actual).all() or frame.actual.lt(0).any():
        raise ValueError("Invalid actual observations")


def analyze(outputs, config):
    frame, status = outputs["forecasts"], outputs["fit_status"]
    ok = frame[frame.status.eq("ok")].copy()
    ok["horizon_band"] = horizon_band(ok.horizon)
    tables = {
        "metrics_available": summarize(ok, ["window", "kind", "model"]),
        "metrics_origin": summarize(ok, ["window", "kind", "model", "origin"]),
        "metrics_series_origin": summarize(ok, ["window", "series_id", "kind", "model", "origin"]),
        "metrics_horizon_available": summarize(ok, ["window", "kind", "model", "horizon"]),
        "metrics_band_available": summarize(ok, ["window", "kind", "model", "horizon_band"]),
        "coverage": status.groupby(["window", "kind", "model", "origin", "status", "reason"], dropna=False).agg(
            fits=("series_id", "count"), warnings=("warning_count", "sum"), runtime_seconds=("runtime_seconds", "sum")).reset_index()}
    coverage = status.groupby(["window", "kind", "model"], as_index=False).agg(
        planned_fits=("status", "size"), successful_fits=("status", lambda s: int(s.eq("ok").sum())),
        failed_fits=("status", lambda s: int(s.eq("failed").sum())),
        skipped_fits=("status", lambda s: int(s.eq("skipped").sum())), warnings=("warning_count", "sum"))
    coverage["success_fraction"] = coverage.successful_fits / coverage.planned_fits
    coverage["planned_targets"] = coverage.planned_fits * config.horizon
    coverage["predicted_targets"] = coverage.successful_fits * config.horizon
    tables["coverage_summary"] = coverage
    panels = {"configured": tuple(config.models), "simple": ("naive", "seasonal_naive", "moving_average"),
              "core_10": tuple(m for m in config.models if m not in ("arima", "sarima")),
              "arima_capped": ("naive", "seasonal_naive", "arima"),
              "sarima_capped": ("naive", "seasonal_naive", "sarima")}
    common_tables, members, origins, horizons, bands, segments, abc = [], [], [], [], [], [], []
    for name, models in panels.items():
        if not models or not set(models).issubset(config.models):
            continue
        common = common_cases(frame, models)
        if common.empty:
            continue
        common["horizon_band"] = horizon_band(common.horizon)
        for collection, groups in ((common_tables, ["window", "kind", "model"]),
                                   (origins, ["window", "kind", "model", "origin"]),
                                   (horizons, ["window", "kind", "model", "horizon"]),
                                   (bands, ["window", "kind", "model", "horizon_band"]),
                                   (segments, ["window", "kind", "regime", "model"]),
                                   (abc, ["window", "kind", "abc_xyz", "model"])):
            measured = summarize(common, groups).assign(comparison=name)
            rank_groups = [column for column in groups if column != "model"]
            for metric in ("mae", "wape", "macro_rmsse"):
                measured[f"rank_{metric}"] = measured.groupby(rank_groups, dropna=False)[metric].rank(method="min")
            collection.append(measured)
        members.append(common[CASE + ["kind"]].drop_duplicates().assign(comparison=name))
    for name, collection in (("metrics_common", common_tables), ("comparison_membership", members),
                             ("metrics_common_origin", origins), ("metrics_common_horizon", horizons),
                             ("metrics_common_band", bands), ("metrics_common_segments", segments), ("metrics_common_abc_xyz", abc)):
        tables[name] = pd.concat(collection, ignore_index=True) if collection else pd.DataFrame()
    paired = paired_losses(frame)
    tables["paired_losses"] = paired
    pair_groups = ["window", "kind", "model", "baseline", "origin"]
    if not paired.empty:
        per_origin = paired.groupby(pair_groups, as_index=False).agg(
            observations=("horizon", "size"), series=("series_id", "nunique"),
            mean_absolute_difference=("absolute_loss_difference", "mean"),
            mean_squared_difference=("squared_loss_difference", "mean"))
        tables["paired_origin"] = per_origin
        tables["paired_variability"] = per_origin.groupby(pair_groups[:-1], as_index=False).agg(
            origins=("origin", "nunique"), mean_difference=("mean_absolute_difference", "mean"),
            minimum_difference=("mean_absolute_difference", "min"), maximum_difference=("mean_absolute_difference", "max"),
            std_difference=("mean_absolute_difference", lambda x: float(np.std(x, ddof=0))),
            candidate_better_origins=("mean_absolute_difference", lambda x: int((x < 0).sum())))
    else:
        tables["paired_origin"], tables["paired_variability"] = pd.DataFrame(), pd.DataFrame()
    # Window sensitivity intersects successes across BOTH windows before comparison.
    window_keys = ["series_id", "origin", "target_date", "horizon", "model"]
    if set(config.windows) == {"expanding", "sliding"}:
        counts = ok.groupby(window_keys).window.nunique()
        shared = counts[counts.eq(2)].reset_index()[window_keys]
        matched = ok.merge(shared, on=window_keys, validate="many_to_one")
        tables["window_sensitivity"] = summarize(matched, ["window", "kind", "model"])
        tables["window_membership"] = shared
    return tables


def run(root, config):
    started = perf_counter()
    root, source, warehouse, env = settings(root)
    build_artifacts(root, source, warehouse)
    sources = [source, root / "data/interim/sales_long.parquet", *sorted((root / "data/raw").glob("*.csv"))]
    before = {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}
    with connection(warehouse, env) as con:
        frames, cohort, schedule, labels, products, label_config = extract(con, root, config)
        source_rows = con.execute("select count(*) from main.fact_daily_sales").fetchone()[0]
        resources = {"duckdb_threads": con.execute("select current_setting('threads')").fetchone()[0],
                     "duckdb_memory_limit": con.execute("select current_setting('memory_limit')").fetchone()[0]}
    extracted_seconds = perf_counter() - started
    outputs, caps = execute(frames, cohort, schedule, config)
    outputs.update(analyze(outputs, config))
    outputs.update(cohort=cohort, origin_calendar=schedule, training_store_item_labels=labels, training_product_labels=products)
    after = {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}
    if before != after:
        raise ValueError("Raw/interim/processed input modification detected")
    target = root / "data/backtests" / f"{schedule.origin.min():%Y%m%d}_{schedule.origin.max():%Y%m%d}_h{config.horizon}_s{config.stride}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".backtests-", dir=root / "data") as temporary:
        temporary = Path(temporary)
        (temporary / "tables").mkdir()
        for name, frame in outputs.items():
            frame.to_parquet(temporary / "tables" / f"{name}.parquet", index=False)
        from .report import figures, write_report
        images = figures(outputs, temporary / "figures")
        statuses = outputs["fit_status"]
        summary = {"status": "passed", "config": asdict(config), "origin_calendar": schedule.to_dict("records"),
                   "cohort": cohort.to_dict("records"), "cohort_fit_end": cohort.segment_fit_end.max(),
                   "segmentation_config": label_config, "expensive_model_caps": caps,
                   "source_rows": source_rows, "input_fingerprints": before, "inputs_unchanged": True,
                   "fit_counts": statuses.status.value_counts().to_dict(), "warnings": int(statuses.warning_count.sum()),
                   "clipped_predictions": int(statuses.clipped_count.sum()),
                   "table_rows": {name: len(frame) for name, frame in outputs.items()},
                   "metrics_common": outputs["metrics_common"].to_dict("records"),
                   "runtime_seconds": perf_counter() - started, "extraction_seconds": extracted_seconds,
                   "resources": resources, "figures": images,
                   "versions": {name: version(name) for name in ("numpy", "pandas", "statsmodels", "scipy", "duckdb", "matplotlib")},
                   "weighting": "No revenue weights: M5 recorded prices lack verified historical availability metadata. Utilities require explicit as-of evidence.",
                   "metric_label": "Selected-cohort metrics; NOT official M5 WRMSSE"}
        write_report(summary, outputs, temporary / "backtest_report.md")
        (temporary / "summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        publish(temporary, target)
    return summary, target


def publish(temporary, target):
    """Replace generated files; prune only obsolete files tracked by our manifest."""
    temporary, target = Path(temporary), Path(target)
    generated = {path.relative_to(temporary).as_posix(): path for path in temporary.rglob("*") if path.is_file()}
    manifest = target / "manifest.json"
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    if not isinstance(previous, list) or any(not isinstance(name, str) for name in previous):
        raise ValueError("Invalid generated-artifact manifest")
    resolved = target.resolve()
    for name in previous:
        if not (target / name).resolve().is_relative_to(resolved):
            raise ValueError("Artifact manifest escapes output directory")
    target.mkdir(parents=True, exist_ok=True)
    for name, path in generated.items():
        final = target / name
        final.parent.mkdir(parents=True, exist_ok=True)
        path.replace(final)
    for name in set(previous) - set(generated):
        obsolete = target / name
        if obsolete.is_file():
            obsolete.unlink()
    manifest.write_text(json.dumps(sorted(generated), indent=2) + "\n", encoding="utf-8")
