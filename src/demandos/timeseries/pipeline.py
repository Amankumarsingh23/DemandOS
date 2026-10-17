"""Small-cohort training-prefix diagnostics and auditable temporal demonstrations."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import numpy as np
import pandas as pd

from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.eda.queries import fetch_frame
from demandos.warehouse.cli import settings, connection, build_artifacts
from .cohort import training_labels, select_cohort
from .diagnostics import (DiagnosticsConfig, eligibility, correlation_diagnostics,
                          stationarity_diagnostics, additive_stl)
from .temporal import (daily_panel, descriptive_lags, difference, fit_location_scale, origin_features,
                       chronological_split, origin_schedule, FeatureEvidence, validate_feature)
from .figures import figures


def extract(con, root, config):
    dates = fetch_frame(con, "select date from main.dim_calendar order by date").date
    split = chronological_split(dates, config.validation_days, config.test_days, config.minimum_observations)
    labels, products, segment_config = training_labels(con, root, split["train_start"], split["train_end"])
    chosen = select_cohort(labels, config.max_store_items)
    con.register("diagnostic_keys", chosen[["store_id", "item_id"]])
    panel = daily_panel(fetch_frame(con, """select f.store_id, f.item_id, f.date, f.units_sold
        from main.fact_daily_sales f join diagnostic_keys k using(store_id,item_id)
        order by f.store_id, f.item_id, f.date"""))
    histories, metadata = {}, []
    for row in chosen.itertuples():
        series_id = f"store_item/{row.store_id}/{row.item_id}"
        history = panel[panel.store_id.eq(row.store_id) & panel.item_id.eq(row.item_id) & panel.date.le(split["train_end"])][["date", "units_sold"]].copy()
        histories[series_id] = history
        metadata.append({"series_id": series_id, "kind": "store_item", "store_id": row.store_id,
                         "item_id": row.item_id, "regime": row.regime, "abc_xyz": row.abc_xyz,
                         "selection_reason": row.selection_reason, "segment_fit_end": split["train_end"]})
    for grain, key, number in (("store", "store_id", config.store_aggregates), ("category", "cat_id", config.category_aggregates)):
        selected = con.execute(f"select {key}, sum(units_sold) as units from eda_{grain}_daily group by {key} order by units desc, {key} limit ?", [number]).fetchall()
        for identifier, _ in selected:
            series_id = f"{grain}/{identifier}"
            histories[series_id] = fetch_frame(con, f"select date, units_sold from eda_{grain}_daily where {key} = ? order by date", [identifier])
            histories[series_id]["date"] = pd.to_datetime(histories[series_id].date)
            metadata.append({"series_id": series_id, "kind": grain, "store_id": identifier if grain == "store" else None,
                             "item_id": None, "regime": None, "abc_xyz": None,
                             "selection_reason": "aggregate_training_volume", "segment_fit_end": split["train_end"]})
    for history in histories.values():
        history["date"] = pd.to_datetime(history.date)
        if history.date.duplicated().any() or not history.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError("Cohort has duplicate/missing dates")
        if history.date.max() > split["train_end"]:
            raise ValueError("Diagnostic history extends past training boundary")
    return histories, pd.DataFrame(metadata), panel, labels, products, segment_config, split, dates


def analyze(histories, config):
    diagnostics, coefficients, observations, decompositions, stl_states = [], [], [], [], []
    for series_id, frame in histories.items():
        frame = frame.copy().reset_index(drop=True)
        frame["series_id"] = series_id
        frame["rolling_mean"] = frame.units_sold.rolling(config.rolling_days, min_periods=1).mean()
        frame["rolling_variance"] = frame.units_sold.rolling(config.rolling_days, min_periods=1).var(ddof=0)
        frame["first_difference"] = difference(frame.units_sold)
        frame["seasonal_difference"] = difference(frame.units_sold, lag=config.seasonal_period)
        variants = {"original": frame.units_sold, "first_difference": frame.first_difference, "seasonal_difference": frame.seasonal_difference}
        if config.include_second_difference:
            frame["second_difference"] = difference(frame.units_sold, order=2)
            variants["second_difference"] = frame.second_difference
        observations.append(frame)
        for variant, values in variants.items():
            # Drop only the deterministic lost prefix, never missing interior dates.
            lost = 0 if variant == "original" else config.seasonal_period if variant == "seasonal_difference" else 2 if variant == "second_difference" else 1
            numeric = values.iloc[lost:].to_numpy(dtype=float)
            correlations = correlation_diagnostics(numeric, config)
            tests = stationarity_diagnostics(numeric, config)
            record = {"series_id": series_id, "variant": variant, "observations": len(numeric), "lost_initial": lost,
                      "mean": float(np.mean(numeric)), "variance": float(np.var(numeric)),
                      "nonzero_fraction": float(np.count_nonzero(numeric) / len(numeric)),
                      "correlation_status": correlations["status"], "correlation_reason": correlations.get("reason"),
                      "correlation_warnings": json.dumps(correlations.get("warnings", [])),
                      "stationarity_status": tests["status"], "stationarity_reason": tests.get("reason"),
                      "interpretation": tests["interpretation"]}
            for name in ("adf", "kpss"):
                result = tests[name] or {}
                for field in ("statistic", "pvalue", "lags"):
                    record[f"{name}_{field}"] = result.get(field)
                record[f"{name}_warnings"] = json.dumps(result.get("warnings", []))
                record[f"{name}_failure"] = result.get("reason")
            for lag in (1, 7, 14, 28):
                coefficient = next((row for row in correlations["coefficients"] if row["lag"] == lag), {})
                record[f"acf_{lag}"] = coefficient.get("acf")
                record[f"pacf_{lag}"] = coefficient.get("pacf")
            diagnostics.append(record)
            coefficients.extend(dict(row, series_id=series_id, variant=variant) for row in correlations["coefficients"])
        reason = eligibility(frame.units_sold, config)
        if reason:
            stl_states.append({"series_id": series_id, "status": "skipped", "reason": reason})
        else:
            fit, reason = additive_stl(frame[["date", "units_sold"]], config.seasonal_period)
            stl_states.append({"series_id": series_id, "status": "skipped" if reason else "ok", "reason": reason})
            if not fit.empty:
                fit["series_id"] = series_id
                np.testing.assert_allclose(fit.trend + fit.seasonal + fit.residual, fit.units_sold, atol=1e-7)
                decompositions.append(fit)
        print(f"Diagnosed {series_id}", flush=True)
    return {"diagnostics": pd.DataFrame(diagnostics),
            "correlations": pd.DataFrame(coefficients, columns=["lag", "acf", "acf_low", "acf_high", "pacf", "pacf_low", "pacf_high", "series_id", "variant"]),
            "transformations": pd.concat(observations, ignore_index=True),
            "stl": pd.concat(decompositions, ignore_index=True) if decompositions else pd.DataFrame(columns=["series_id"]),
            "stl_status": pd.DataFrame(stl_states)}


def temporal_demo(panel, split, dates, config):
    origin = split["train_end"]
    fitted = fit_location_scale(panel, origin)
    features = origin_features(panel, origin, config.horizons, config.lags, config.rolling_days, fitted=fitted)
    prefix = panel[panel.date.le(origin)]
    safe_again = origin_features(prefix, origin, config.horizons, config.lags, config.rolling_days, fitted=fit_location_scale(prefix, origin))
    pd.testing.assert_frame_equal(features, safe_again)
    if features.target_date.max() > split["validation_end"]:
        raise ValueError("Demonstration horizon spills past validation targets")
    leak_date = origin + pd.Timedelta(days=6)
    try:
        validate_feature(FeatureEvidence("target_lag_1_for_h7", None, leak_date, leak_date), origin)
    except ValueError as error:
        rejected = str(error)
    else:
        raise ValueError("Expected h=7 leakage was not rejected")
    initial = int((origin - pd.Timestamp(dates.min())).days) + 1
    return {"origin_features": features, "training_scale": fitted,
            "historical_lags": descriptive_lags(prefix, config.lags, config.rolling_days),
            "expanding_origins": origin_schedule(dates, initial, horizon=28, step=28),
            "rolling_origins": origin_schedule(dates, initial, horizon=28, step=28, training_window=365)}, {
                "origin": origin, "horizon": 7, "target": origin + pd.Timedelta(days=7),
                "naive_target_lag_1_source": leak_date, "naive_status": "rejected", "reason": rejected,
                "correct_sales_at_origin_source": origin, "correct_origin_minus_1_source": origin - pd.Timedelta(days=1),
                "future_invariance": "features and fitted parameters identical with/without all post-origin actuals"}


def run(root, config):
    from .report import write_report

    started = perf_counter()
    root, source, warehouse, env = settings(root)
    build = build_artifacts(root, source, warehouse)
    with connection(warehouse, env) as con:
        current = con.execute("select count(*), sum(units_sold) from main.fact_daily_sales").fetchone()
        baseline = json.loads((warehouse / "analytics_quality_report.json").read_text())["fact"]
        if current != (baseline["daily_rows"], baseline["units_sold"]):
            raise ValueError("Warehouse baseline does not reconcile")
        histories, metadata, panel, labels, products, label_config, split, dates = extract(con, root, config)
    outputs = analyze(histories, config)
    examples, leakage = temporal_demo(panel, split, dates, config)
    outputs.update(examples)
    outputs.update(cohort=metadata, training_store_item_labels=labels, training_product_labels=products)
    target = root / "data/diagnostics" / f"train_{split['train_end']:%Y%m%d}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".diagnostics-", dir=root / "data") as temporary:
        temporary = Path(temporary)
        (temporary / "tables").mkdir()
        for name, frame in outputs.items():
            frame.to_parquet(temporary / "tables" / f"{name}.parquet", index=False)
        paths = figures(outputs["transformations"], outputs["correlations"], outputs["stl"], metadata,
                        temporary / "figures", config, split)
        summary = {
            "status": "passed", "config": asdict(config), "split": split, "source": fingerprint(source),
            "source_rows": int(current[0]), "source_units": int(current[1]),
            "selection": "illustrative, deterministic training-only Season 4 regimes and ABC/XYZ; not a representative sample",
            "segmentation_config": asdict(label_config), "cohort": metadata.to_dict("records"),
            "diagnostics": outputs["diagnostics"].to_dict("records"), "stl": outputs["stl_status"].to_dict("records"),
            "leakage_demonstration": leakage, "horizons_verified": list(config.horizons),
            "table_rows": {key: len(value) for key, value in outputs.items()}, "figures": paths,
            "warehouse_build_tests_passed": sum(r["unique_id"].startswith("test.") and r["status"] == "pass" for r in build["results"]),
            "versions": {name: version(name) for name in ("duckdb", "numpy", "pandas", "statsmodels", "matplotlib")},
            "runtime_seconds": perf_counter() - started,
            "warnings": ["All diagnostics and segment fits stop at train_end; no holdout targets used for fitting.",
                         "P-values and correlation intervals are exploratory, uncorrected for multiple comparisons and not mathematical stationarity verdicts.",
                         "Sparse/constant/short histories are skipped, never assigned fabricated estimates.",
                         "STL and trailing plots are descriptive; centered/full-series components are forbidden as forecasting features.",
                         "Recorded prices have no verified known-ahead availability and are excluded from origin feature demos.",
                         "Origin schedules are boundary demonstrations, not trained forecasts or a backtesting engine."]}
        write_report(summary, temporary / "diagnostic_report.md")
        (temporary / "summary.json").write_text(json.dumps(json_ready(summary), indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
        target.mkdir(parents=True, exist_ok=True)
        for path in temporary.rglob("*"):
            if path.is_file():
                final = target / path.relative_to(temporary)
                final.parent.mkdir(parents=True, exist_ok=True)
                path.replace(final)
    return summary, target


def load_config(path):
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    if set(data) != {"timeseries"}:
        raise ValueError("Expected [timeseries] configuration")
    return DiagnosticsConfig(**data["timeseries"])
