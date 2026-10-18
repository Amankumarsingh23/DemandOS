"""Verified warehouse -> compact origin reductions -> bounded direct-horizon tables."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd

from demandos.backtesting.design import load_config as load_backtest_config, schedules
from demandos.backtesting.pipeline import publish
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.eda.queries import fetch_frame
from demandos.timeseries.cohort import training_labels, select_cohort
from demandos.warehouse.cli import settings, connection, build_artifacts
from .contracts import (feature_dictionary, origin_dictionary, label_dictionary, predictive_columns,
                        write_parquet, KEYS)
from .encoding import fit_categories
from .sql import origin_features, validate_price_evidence
from .validation import generate


def source_views(con):
    con.execute("""create or replace temp view feature_base_sales as
        select store_id,item_id,date,units_sold from main.fact_daily_sales""")
    con.execute("create or replace temp view feature_sales as select * from feature_base_sales")
    con.execute("create or replace temp view feature_prices as select * from main.dim_price_week")
    con.execute("create or replace temp view feature_calendar as select * from main.dim_calendar")
    invalid = con.execute("""select count(*) from feature_calendar where
        (event_name_1 is null)<>(event_type_1 is null) or (event_name_2 is null)<>(event_type_2 is null)
        or event_type_1 not in ('Cultural','National','Religious','Sporting')
        or event_type_2 not in ('Cultural','National','Religious','Sporting')
        or snap_CA not in (0,1) or snap_TX not in (0,1) or snap_WI not in (0,1)""").fetchone()[0]
    if invalid:
        raise ValueError("Invalid event or SNAP calendar metadata")


def load_keys(con, cutoff, selected=None):
    """Static dimensions with first observed date established by the allowed cutoff."""
    join = ""
    if selected is not None:
        if selected.empty or not {"store_id", "item_id"}.issubset(selected) or selected[["store_id", "item_id"]].isna().any().any() or selected.duplicated(["store_id", "item_id"]).any():
            raise ValueError("Unique nonempty explicit store-item keys required")
        con.register("feature_requested_keys", selected[["store_id", "item_id"]])
        join = "join feature_requested_keys r using(store_id,item_id)"
    frame = fetch_frame(con, f"""with first as (select f.store_id,f.item_id,min(f.date) as first_observed_date
        from feature_base_sales f {join} where f.date<=? group by f.store_id,f.item_id)
        select first.*,s.state_id,p.dept_id,p.cat_id from first
        left join main.dim_store s using(store_id) left join main.dim_product p using(item_id)
        order by store_id,item_id""", [pd.Timestamp(cutoff).date()])
    if frame.empty or frame.isna().any().any() or frame.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Canonical dimension joins missing or multiplying keys")
    if selected is not None and len(frame) != len(selected):
        raise ValueError("Requested series absent at the allowed selection cutoff")
    return frame


def choose_cohort(con, root, config, explicit=None):
    if explicit is None:
        first = pd.Timestamp(con.execute("select min(date) from feature_calendar").fetchone()[0])
        labels, _, _ = training_labels(con, root, first, pd.Timestamp(config.mapping_fit_end))
        selected = select_cohort(labels, config.max_store_items)
    else:
        selected = explicit.copy()
        if set(selected) != {"store_id", "item_id"}:
            raise ValueError("Explicit cohort file must contain only store_id,item_id")
        selected["selection_reason"] = "explicit_cohort"
    if len(selected) > config.max_expanded_series:
        raise ValueError("Selected cohort exceeds configured horizon-expansion cap")
    keys = load_keys(con, config.mapping_fit_end, selected)
    metadata = selected[[column for column in ("store_id", "item_id", "regime", "abc_xyz", "selection_reason", "segment_fit_end") if column in selected]]
    return keys, keys.merge(metadata, on=["store_id", "item_id"], validate="one_to_one")


def fit_mapping(con, keys, cutoff):
    con.register("feature_mapping_keys", keys[["store_id", "item_id"]])
    frame = fetch_frame(con, """select f.store_id,s.state_id,f.item_id,p.dept_id,p.cat_id,max(f.date) as date
        from main.fact_daily_sales f join feature_mapping_keys k using(store_id,item_id)
        join main.dim_store s using(store_id) join main.dim_product p using(item_id)
        where f.date<=? and f.units_sold>0 group by all""", [pd.Timestamp(cutoff).date()])
    return fit_categories(frame, cutoff)


def totals(con):
    row = con.execute("""select count(*),sum(units_sold),sum(observed_revenue),
        count(distinct (store_id,item_id)) from main.fact_daily_sales""").fetchone()
    return {"daily_rows": row[0], "units_sold": row[1], "known_revenue": row[2], "series": row[3]}


def check_design(con, root, config):
    dates = fetch_frame(con, "select date from feature_calendar order by date").date
    backtest = load_backtest_config(root / "configs/backtesting.toml")
    schedule = schedules(dates, backtest).drop_duplicates("origin")
    evaluation = pd.DatetimeIndex(schedule.origin).sort_values()
    training = pd.DatetimeIndex(pd.to_datetime(config.training_origins))
    if pd.Timestamp(config.mapping_fit_end) >= min(training.min(), evaluation.min()):
        raise ValueError("Mapping/selection cutoff must precede all origins")
    if training.max() + pd.Timedelta(days=28) > evaluation.min():
        raise ValueError("Training labels overlap Season 7 evaluation target periods")
    if backtest.horizon != 28:
        raise ValueError("Season 8 demonstration requires Season 7 horizon=28")
    for origin in training:
        if origin not in set(pd.to_datetime(dates)) or origin + pd.Timedelta(days=28) not in set(pd.to_datetime(dates)):
            raise ValueError("Training origin/targets outside available calendar")
    return training, evaluation, schedule


def invariance_checks(con, origins, config, mapping, evidence=None):
    """Change all post-origin actuals and future-week prices via read-only temp views."""
    results = []
    for origin in origins:
        origin = pd.Timestamp(origin)
        verified = validate_price_evidence(con, evidence)
        baseline = generate(con, [origin], config, mapping, verified_prices=verified)
        # Dates are validated Timestamps, never untrusted SQL text.
        day = origin.date().isoformat()
        con.execute(f"""create or replace temp view feature_sales as select store_id,item_id,date,
            case when date>date '{day}' then units_sold+1000000 else units_sold end as units_sold
            from feature_base_sales""")
        con.execute(f"""create or replace temp view feature_prices as select p.* replace(
            case when p.wm_yr_wk in (select wm_yr_wk from feature_calendar group by wm_yr_wk
                having min(date)>date '{day}') then p.sell_price*100+999 else p.sell_price end as sell_price)
            from main.dim_price_week p""")
        changed = generate(con, [origin], config, mapping, verified_prices=validate_price_evidence(con, evidence))
        pd.testing.assert_frame_equal(baseline.features, changed.features)
        con.execute("create or replace temp view feature_sales as select * from feature_base_sales")
        con.execute("create or replace temp view feature_prices as select * from main.dim_price_week")
        # Independently prove inference can run without any future actuals at all.
        con.execute(f"create or replace temp view feature_sales as select * from feature_base_sales where date<=date '{day}'")
        prefix = generate(con, [origin], config, mapping, verified_prices=verified)
        pd.testing.assert_frame_equal(baseline.features, prefix.features)
        con.execute("create or replace temp view feature_sales as select * from feature_base_sales")
        results.append({"origin": origin, "rows_compared": len(baseline.features), "all_28_horizons_identical": True,
                        "future_actuals_removed_identical": True,
                        "future_price_check": "excluded by default" if evidence is None else "verified historical as-of prices only"})
    return results


def read_table(path):
    return pd.read_parquet(path) if Path(path).suffix.lower() == ".parquet" else pd.read_csv(path)


def run(root, config, *, inference_only=False, cohort_path=None, price_evidence_path=None):
    started = perf_counter()
    root, source, warehouse, env = settings(root)
    build = build_artifacts(root, source, warehouse)
    inputs = [source, root / "data/interim/sales_long.parquet", *sorted((root / "data/raw").glob("*.csv"))]
    inputs += [Path(path).resolve() for path in (cohort_path, price_evidence_path) if path is not None]
    before = {str(path): fingerprint(path) for path in inputs}
    explicit = read_table(cohort_path) if cohort_path else None
    evidence = read_table(price_evidence_path) if price_evidence_path else None
    target = root / "data/features" / ("inference" if inference_only else "direct_h28")
    with connection(warehouse, env) as con, TemporaryDirectory(prefix=".features-", dir=root / "data") as temporary:
        temporary = Path(temporary)
        source_views(con)
        training_origins, evaluation_origins, schedule = check_design(con, root, config)
        initial = totals(con)
        baseline = json.loads((warehouse / "analytics_quality_report.json").read_text(encoding="utf-8"))["fact"]
        for name in initial:
            if str(initial[name]) != str(baseline[name]):
                raise ValueError(f"Warehouse baseline changed: {name}")
        full_report = {"executed": False}
        if config.full_origin_pass:
            full_keys = load_keys(con, evaluation_origins.min())
            con.register("feature_keys", full_keys)
            full_started = perf_counter()
            full = origin_features(con, evaluation_origins.min(), config, validate_price_evidence(con, evidence))
            if len(full) != initial["series"]:
                raise ValueError("Full origin pass does not cover every observed series")
            write_parquet(full, temporary / "origin_level/full_origin.parquet", origin_dictionary(config))
            full_report = {"executed": True, "origin": evaluation_origins.min(), "rows": len(full),
                           "bounded_history_rows": int(full.history_observations.sum()),
                           "dataframe_bytes": int(full.memory_usage(deep=True).sum()),
                           "seconds": perf_counter() - full_started,
                           "null_counts": {name: int(value) for name, value in full.isna().sum().items()}}
            del full, full_keys
        keys, cohort = choose_cohort(con, root, config, explicit)
        con.register("feature_keys", keys)
        mapping = fit_mapping(con, keys, config.mapping_fit_end)
        verified = validate_price_evidence(con, evidence)
        evaluations = generate(con, evaluation_origins, config, mapping, verified_prices=verified)
        trainings = (None if inference_only else generate(con, training_origins, config, mapping,
            inference_only=False, label_cutoff=evaluation_origins.min(), verified_prices=verified))
        invariance = invariance_checks(con, evaluation_origins, config, mapping, evidence)
        for role, batch in (("evaluation", evaluations), ("training", trainings)):
            if batch is None:
                continue
            write_parquet(batch.features, temporary / f"{role}_features.parquet", feature_dictionary(config))
            if batch.labels is not None:
                write_parquet(batch.labels, temporary / f"{role}_labels.parquet", label_dictionary())
            for origin, rows in batch.features.groupby("origin_date"):
                write_parquet(rows, temporary / f"by_origin/{role}_{origin:%Y%m%d}.parquet", feature_dictionary(config))
            origin_frame = batch.features[[entry["name"] for entry in origin_dictionary(config)]].drop_duplicates()
            write_parquet(origin_frame, temporary / f"origin_level/{role}_origins.parquet", origin_dictionary(config))
        cohort.to_csv(temporary / "cohort.csv", index=False)
        samples = evaluations.features.groupby(["store_id", "origin_date"], sort=True).head(1).head(9)
        samples.to_csv(temporary / "sample_features.csv", index=False)
        final = totals(con)
        if final != initial:
            raise ValueError("Warehouse sales/revenue totals changed")
        after = {str(path): fingerprint(path) for path in inputs}
        if before != after:
            raise ValueError("Raw/interim/processed inputs were modified")
        dictionary = {"origin": origin_dictionary(config), "features": feature_dictionary(config), "labels": label_dictionary(),
                      "predictive_columns": predictive_columns(config)}
        (temporary / "feature_dictionary.json").write_text(json.dumps(dictionary, indent=2) + "\n", encoding="utf-8")
        (temporary / "category_mappings.json").write_text(json.dumps(mapping.as_dict(), indent=2) + "\n", encoding="utf-8")
        summary = {"status": "passed", "config": asdict(config), "inference_only": inference_only,
                   "grain": KEYS, "origin_contract": "End of day, sales through origin inclusive",
                   "full_origin_pass": full_report, "cohort_series": len(keys), "cohort_fit_end": config.mapping_fit_end,
                   "evaluation": evaluations.validation, "training": trainings.validation if trainings else None,
                   "training_labels": len(trainings.labels) if trainings else 0,
                   "evaluation_origins": list(evaluation_origins), "training_origins": list(training_origins),
                   "label_cutoff": evaluation_origins.min(), "invariance": invariance,
                   "predictive_columns": predictive_columns(config), "source_totals": initial,
                   "input_fingerprints": before, "inputs_unchanged": True, "warehouse_totals_unchanged": True,
                   "historical_price_policy": "unavailable without explicit observed_at and available_at evidence" if evidence is None else "explicit verified historical evidence",
                   "known_ahead_calendar_assumption": "Public dates/events/SNAP flags treated as scheduled and known at origin; no archive verifies announcement times",
                   "dbt_build_tests_passed": sum(r["unique_id"].startswith("test.") and r["status"] == "pass" for r in build["results"]),
                   "automated_test_results": "See separately recorded test_results.json; CLI does not rerun unit/dbt suites",
                   "resources": {"duckdb_threads": con.execute("select current_setting('threads')").fetchone()[0],
                                 "duckdb_memory_limit": con.execute("select current_setting('memory_limit')").fetchone()[0],
                                 "evaluation_dataframe_bytes": int(evaluations.features.memory_usage(deep=True).sum())},
                   "runtime_seconds": perf_counter() - started,
                   "versions": {name: version(name) for name in ("pandas", "numpy", "pyarrow", "duckdb")}}
        from .report import write_report, write_dictionary
        write_report(summary, samples, temporary / "feature_report.md")
        write_dictionary(dictionary, temporary / "feature_dictionary.md")
        (temporary / "validation_summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        publish(temporary, target)
    return summary, target
