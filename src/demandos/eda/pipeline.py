"""Reproducible EDA from an existing, verified Season 3 warehouse."""

from dataclasses import asdict
from datetime import date
from decimal import Decimal
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd

from demandos.data.pipeline import fingerprint
from demandos.warehouse.cli import build_artifacts, connection, settings
from .analysis import (abc_xyz_matrix, decomposition_examples, product_segments, segment_totals,
                       select_series, series_segments, temporal_tables, validate_totals)
from .figures import generate_figures
from .queries import create_views, expand_quantiles, fetch_frame, group_summary
from .statistics import outlier_flags


def json_ready(value):
    """Represent undefined statistics as JSON null, never nonstandard NaN tokens."""
    import math
    import numbers

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        return str(value) if value.is_finite() else None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else None
    return value


def collect(con, root, config):
    bounds = con.execute("select cast(min(date) as date), cast(max(date) as date) from main.dim_calendar").fetchone()
    start = date.fromisoformat(config.start_date) if config.start_date else bounds[0]
    end = date.fromisoformat(config.end_date) if config.end_date else bounds[1]
    if not bounds[0] <= start <= end <= bounds[1]:
        raise ValueError("Selected analysis window must lie within observed history")
    create_views(con, start, end)
    baseline = fetch_frame(con, """select count(*) as daily_rows, count(distinct date) as observed_days,
        sum(units_sold) as units_sold, sum(observed_revenue) as known_revenue,
        count(*) filter (where units_sold = 0) as zero_sales_count,
        count(*) filter (where sell_price is null) as missing_price_count,
        count(*) filter (where observed_revenue is null) as unknown_revenue_count
        from eda_fact""").iloc[0].to_dict()
    if not baseline["daily_rows"] or baseline["observed_days"] != (end - start).days + 1:
        raise ValueError("Empty or noncontinuous analysis window")
    frames = {}
    for grain, key in (("store", "store_id"), ("state", "state_id"), ("category", "cat_id"), ("product", "item_id")):
        frames[grain] = expand_quantiles(group_summary(con, grain, key))
        frames[grain]["volume_rank"] = frames[grain].units_sold.rank(method="dense", ascending=False).astype(int)
    statement = (root / "sql/eda/store_item_statistics.sql").read_text(encoding="utf-8")
    series = []
    for store in frames["store"].store_id:
        series.append(expand_quantiles(fetch_frame(con, statement, [store])))
        print(f"EDA statistics: {store}", flush=True)
    frames["store_item"] = series_segments(pd.concat(series, ignore_index=True), config)
    hierarchy = fetch_frame(con, "select * from main.dim_product")
    stores = fetch_frame(con, "select * from main.dim_store")
    frames["store_item"] = frames["store_item"].merge(hierarchy, on="item_id", validate="many_to_one").merge(stores, on="store_id", validate="many_to_one")
    frames["store_item"]["rank_within_store"] = frames["store_item"].groupby("store_id").units_sold.rank(method="dense", ascending=False).astype(int)
    frames["store_item"]["rank_within_product"] = frames["store_item"].groupby("item_id").units_sold.rank(method="dense", ascending=False).astype(int)
    weekly = fetch_frame(con, (root / "sql/eda/weekly_product_statistics.sql").read_text(encoding="utf-8"))
    frames["product_segments"] = product_segments(frames["product"], weekly, config)
    revenue = baseline["known_revenue"] or Decimal(0)
    frames["abc_xyz_matrix"], frames["abc_xyz_unclassified"] = abc_xyz_matrix(frames["product_segments"], baseline["units_sold"], revenue)
    frames["abc_summary"] = segment_totals(frames["product_segments"], "abc_class", "products", baseline["units_sold"], revenue)
    frames["regime_summary"] = segment_totals(frames["store_item"], "regime", "series", baseline["units_sold"], revenue)
    for name, keys in (("regime_subgroups", ["state_id", "cat_id", "regime"]), ("regime_by_store", ["store_id", "regime"])):
        frame = frames["store_item"].groupby(keys).agg(series=("item_id", "size"), units_sold=("units_sold", "sum")).reset_index()
        group_keys = keys[:-1]
        frame["series_share_within_group"] = frame.series / frame.groupby(group_keys).series.transform("sum")
        frame["volume_share_within_group"] = frame.units_sold / frame.groupby(group_keys).units_sold.transform("sum")
        frames[name] = frame
    daily = fetch_frame(con, """select o.*, c.* exclude(date)
        from eda_overall_daily o join eda_calendar c using(date) order by date""")
    store_daily = fetch_frame(con, "select * from eda_store_daily order by store_id, date")
    category_daily = fetch_frame(con, "select * from eda_category_daily order by cat_id, date")
    frames.update(temporal_tables(daily, store_daily, category_daily))
    frames["category_daily"] = category_daily
    frames["store_daily"] = store_daily
    frames["state_daily"] = fetch_frame(con, "select * from eda_state_daily order by state_id, date")
    frames["decomposition"] = decomposition_examples(daily, category_daily)
    chosen = select_series(frames["store_item"], config)
    frames["selected_series"] = chosen
    con.register("eda_selected_keys", chosen)
    selected = fetch_frame(con, """select f.*, s.selection, c.event_name_1, c.event_name_2
        from eda_fact f join eda_selected_keys s using(store_id, item_id)
        join eda_calendar c using(date) order by store_id, item_id, date""")
    anomalies = [outlier_flags(group, config) for _, group in selected.groupby(["store_id", "item_id"], sort=True)]
    frames["selected_anomalies"] = pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()
    validate_totals(frames, baseline)
    if int(frames["daily"].units_sold.sum()) != baseline["units_sold"]:
        raise ValueError("Temporal aggregation mismatch")
    if len(frames["selected_anomalies"]) != len(selected):
        raise ValueError("Anomaly screening changed observation count")
    full = (start, end) == bounds
    return frames, baseline, start, end, full


def run(root, config):
    from .report import write_report

    started = perf_counter()
    root, source, warehouse, env = settings(root)
    artifacts = build_artifacts(root, source, warehouse)
    if any(row["status"] != "pass" for row in artifacts["results"] if row["unique_id"].startswith("test.")):
        raise ValueError("Warehouse build has failing tests")
    with connection(warehouse, env) as con:
        frames, baseline, start, end, full = collect(con, root, config)
    if full:
        previous = json.loads((warehouse / "analytics_quality_report.json").read_text())
        for key, value in baseline.items():
            if key in previous["fact"] and str(value) != str(previous["fact"][key]):
                raise ValueError(f"Season 3 reconciliation failed: {key}")
    target = root / "data/eda" / f"{start:%Y%m%d}_{end:%Y%m%d}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".eda-", dir=root / "data") as temporary:
        temporary = Path(temporary)
        (temporary / "tables").mkdir()
        for name, frame in frames.items():
            frame.to_parquet(temporary / "tables" / f"{name}.parquet", index=False)
        charts = generate_figures(frames, temporary / "figures", config, f"{start} to {end}")
        report = {
            "status": "passed", "window_start": str(start), "window_end": str(end), "full_history": full,
            "baseline": baseline, "config": asdict(config), "source": fingerprint(source),
            "reconciliation": "all five summary grains preserve rows, units, known revenue, zeros, missing prices and unknown revenue",
            "warehouse_build_tests_passed": sum(r["unique_id"].startswith("test.") and r["status"] == "pass" for r in artifacts["results"]),
            "abc": frames["abc_summary"].to_dict("records"),
            "abc_xyz": frames["abc_xyz_matrix"].to_dict("records"),
            "abc_xyz_unclassified": frames["abc_xyz_unclassified"].to_dict("records"),
            "regimes": frames["regime_summary"].to_dict("records"),
            "xyz_reasons": frames["product_segments"].xyz_reason.value_counts().to_dict(),
            "complete_weeks": int(frames["weekly"].observed_days.eq(7).sum()),
            "partial_weeks_excluded": int(frames["weekly"].observed_days.ne(7).sum()),
            "leading_zero_series": int(frames["store_item"].leading_zero_days.gt(0).sum()),
            "first_positive_sensitivity_changed": int(frames["store_item"].first_positive_sensitivity_changed.sum()),
            "selected_anomaly_rows": len(frames["selected_anomalies"]),
            "iqr_flags_selected": int(frames["selected_anomalies"].iqr_flag.sum()) if len(frames["selected_anomalies"]) else 0,
            "robust_flags_selected": int(frames["selected_anomalies"].robust_flag.sum()) if len(frames["selected_anomalies"]) else 0,
            "table_rows": {name: len(frame) for name, frame in frames.items()}, "figures": charts,
            "versions": {name: version(name) for name in ("duckdb", "pandas", "matplotlib", "statsmodels")},
            "runtime_seconds": perf_counter() - started,
            "warnings": ["Recorded sales are not uncensored demand; zeros and first positive sales do not prove stock status or launch dates.",
                         "ABC and XYZ describe the selected window. Recompute using training data only for forecasting experiments.",
                         "Full-window additive STL and global IQR flags are descriptive, not forecasting features or proven data errors.",
                         "Event associations and pooled monthly patterns are confounded by trend, assortment and other unmeasured factors.",
                         "Revenue uses historical M5 selling prices in US-dollar context, not profit or verified company accounting outcomes."],
        }
        write_report(frames, report, temporary / "business_report.md")
        (temporary / "quality_report.json").write_text(json.dumps(json_ready(report), indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
        target.mkdir(parents=True, exist_ok=True)
        # Publish only a fully computed run; individual replacements are not a transaction.
        for path in temporary.rglob("*"):
            if path.is_file():
                destination = target / path.relative_to(temporary)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
    return report, target
