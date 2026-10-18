"""One fixed development protocol, origin-specific global refits and honest scoring."""

from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from demandos.backtesting.pipeline import publish
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.features.contracts import KEYS
from demandos.features.pipeline import totals
from demandos.features.validation import generate
from demandos.forecasting.evaluation import metrics
from demandos.warehouse.cli import settings, connection, build_artifacts
from .config import LIBRARIES
from .models import GlobalModel, TARGET
from .dataset import build, eligible_rows, row_digest
from .evaluation import (BASELINES, EVENT_FEATURES, statistical_predictions, attach_targets,
                         comparison_tables, permutation_importance)


def fit_and_record(model, training, directory, fit_id, phase, origin, records,
                   *, validation=None, validation_end=None, probe=None):
    """A failed fitting attempt is explicit; no statistical-model fallback is substituted."""
    start = perf_counter()
    try:
        model.fit(training, allowed_cutoff=origin, validation=validation, validation_cutoff=validation_end)
        model.save(directory)
        if probe is not None:
            reloaded = GlobalModel.load(directory)
            np.testing.assert_allclose(model.predict(probe).prediction, reloaded.predict(probe).prediction, rtol=1e-10, atol=1e-10)
        warnings_count = len(model.metadata["warnings"]) + len(model.metadata["native_warning_lines"])
        model_bytes = sum(path.stat().st_size for path in Path(directory).iterdir() if path.name.startswith("model."))
        records.append({"fit_id": fit_id, "model": model.library, "origin": origin, "phase": phase, "status": "ok", "reason": None,
                        "runtime_seconds": model.metadata["fit_seconds"], "rows": len(training),
                        "series": model.metadata["training_series"], "rounds": model.rounds,
                        "warning_count": warnings_count, "model_bytes": model_bytes, "save_load_equal": probe is not None,
                        "training_target_max": model.metadata["training_target_max"]})
        return model
    except Exception as error:
        records.append({"fit_id": fit_id, "model": model.library, "origin": origin, "phase": phase,
                        "status": "failed", "reason": f"{type(error).__name__}: {error}",
                        "runtime_seconds": perf_counter() - start, "rows": len(training), "series": None,
                        "warning_count": None, "rounds": None, "save_load_equal": False})
        raise RuntimeError(f"Required {phase} {model.library} fit failed; publication blocked: {error}") from error


def future_checks(con, data, directory):
    records = []
    con.register("feature_keys", data["evaluation_keys"])
    for origin in data["origins"]:
        day = origin.date().isoformat()
        before = data["inference"].loc[data["inference"].origin_date.eq(origin)].reset_index(drop=True)
        con.execute(f"""create or replace temp view feature_sales as select store_id,item_id,date,
            case when date>date '{day}' then units_sold+1000000 else units_sold end as units_sold from global_daily_panel""")
        con.execute(f"""create or replace temp view feature_prices as select p.* replace(
            case when wm_yr_wk in (select wm_yr_wk from feature_calendar group by wm_yr_wk having min(date)>date '{day}')
            then sell_price*100+999 else sell_price end as sell_price) from main.dim_price_week p""")
        after = generate(con, [origin], data["feature_config"], data["mapping"]).features.reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)
        con.execute("create or replace temp view feature_sales as select * from global_daily_panel")
        con.execute("create or replace temp view feature_prices as select * from main.dim_price_week")
        changed = data["pool"].copy(deep=False)
        future = pd.to_datetime(changed.target_date).gt(origin)
        changed[TARGET] = changed[TARGET].where(~future, changed[TARGET] + 1000000)
        allowed_before, allowed_after = eligible_rows(data["pool"], origin), eligible_rows(changed, origin)
        pd.testing.assert_frame_equal(allowed_before, allowed_after)
        digest = row_digest(allowed_before)
        for library in LIBRARIES:
            model = GlobalModel.load(directory / "models" / library / f"origin_{origin:%Y%m%d}")
            first, second = model.predict(before), model.predict(after)
            np.testing.assert_allclose(first.prediction, second.prediction, rtol=0, atol=1e-12)
            records.append({"model": library, "origin": origin, "rows": len(first),
                            "future_sales_predictors_unchanged": True, "future_price_exclusion_unchanged": True,
                            "forecast_unchanged": True, "future_labels_fit_inputs_unchanged": True,
                            "eligible_fit_digest": digest})
    return pd.DataFrame(records)


def run(root, config):
    started = perf_counter()
    root, source, warehouse, env = settings(root)
    build_artifacts(root, source, warehouse)
    sources = [source, root / "data/interim/sales_long.parquet", *sorted((root / "data/raw").glob("*.csv"))]
    before = {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}
    target = root / "data/boosting/global_h28"
    with connection(warehouse, env) as con, TemporaryDirectory(prefix=".boosting-", dir=root / "data") as temp:
        temporary = Path(temp)
        initial = totals(con)
        expected = json.loads((warehouse / "analytics_quality_report.json").read_text())["fact"]
        if any(str(value) != str(expected[name]) for name, value in initial.items()):
            raise ValueError("Warehouse totals do not match verified baseline")
        data = build(con, root, config)
        build_seconds = perf_counter() - started
        tables, fit_records, predictions, coverage, selected_rounds, permutation = {}, [], [], [], {}, []
        dev_train, dev_val = data["development_train"], data["development_validation"]
        print(f"Historical pool: {len(data['pool']):,} rows / {config.training_series} series; development {len(dev_train):,}+{len(dev_val):,}", flush=True)
        validation_scores, dev_importance = [], []
        for library in LIBRARIES:
            model = GlobalModel(library, data["feature_columns"], config, data["feature_config"])
            path = temporary / "models" / library / "development"
            model = fit_and_record(model, dev_train, path, f"{library}/development", "development",
                                   pd.Timestamp(config.validation_origin), fit_records, validation=dev_val,
                                   validation_end=data["validation_end"], probe=dev_val.drop(columns=TARGET))
            selected_rounds[library] = model.rounds
            result = model.predict(dev_val.drop(columns=TARGET))
            # Key join, never rely on implicit row-order equality for labels.
            joined = result.merge(dev_val[KEYS + ["target_date", TARGET]], on=KEYS + ["target_date"], validate="one_to_one")
            validation_scores.append({"model": library, "experiment": "primary", "rounds": model.rounds,
                                      **metrics(joined[TARGET], joined.prediction)})
            dev_importance.append(model.importance().assign(phase="development"))
            permutation.append(permutation_importance(model, dev_val, config))
            if library == "lightgbm":
                ablated_features = [name for name in data["feature_columns"] if name not in EVENT_FEATURES]
                ablated = GlobalModel(library, ablated_features, replace(config, iterations=model.rounds), data["feature_config"])
                ablated = fit_and_record(ablated, dev_train, temporary / "models/lightgbm/ablation_no_events",
                    "lightgbm/ablation_no_events", "validation_only_ablation", pd.Timestamp(config.validation_origin),
                    fit_records, probe=dev_val.drop(columns=TARGET))
                altered = ablated.predict(dev_val.drop(columns=TARGET)).merge(
                    dev_val[KEYS + ["target_date", TARGET]], on=KEYS + ["target_date"], validate="one_to_one")
                validation_scores.append({"model": library, "experiment": "no_events_fixed_rounds", "rounds": ablated.rounds,
                                          **metrics(altered[TARGET], altered.prediction)})
            print(f"Developed {library}: {model.rounds}/{config.iterations} rounds, internal validation only", flush=True)
        del model, dev_train, dev_val
        unknowns, eligibility = [], []
        for origin in data["origins"]:
            training = eligible_rows(data["pool"], origin)
            inference = data["inference"].loc[data["inference"].origin_date.eq(origin)].copy()
            eligibility.append({"origin": origin, "rows": len(training), "series": len(training[["store_id", "item_id"]].drop_duplicates()),
                                "origins": training.origin_date.nunique(), "target_min": training.target_date.min(),
                                "target_max": training.target_date.max(), "fit_digest": row_digest(training)})
            for library in LIBRARIES:
                model = GlobalModel(library, data["feature_columns"], replace(config, iterations=selected_rounds[library]), data["feature_config"])
                fit_id = f"{library}/{origin:%Y%m%d}"
                model = fit_and_record(model, training, temporary / "models" / library / f"origin_{origin:%Y%m%d}",
                    fit_id, "evaluation_refit", origin, fit_records, probe=inference)
                predictions.append(model.predict(inference))
                for member in data["evaluation_cohort"].itertuples():
                    coverage.append({"fit_id": fit_id, "model": library, "origin": origin,
                                     "series_id": member.series_id, "status": "ok", "reason": None})
                unknowns.extend({"model": library, "origin": origin, "feature": name,
                                 "unknown_rows": int((~inference[name].astype(str).isin(model.vocabulary[name])).sum())}
                                for name in model.categories)
                print(f"Fitted {library} {origin:%Y-%m-%d}: {len(training):,} labels observed by origin", flush=True)
        baseline_predictions, baseline_fits, baseline_coverage = statistical_predictions(data)
        predictions.append(baseline_predictions)
        fit_records.extend(baseline_fits)
        coverage.extend(baseline_coverage)
        names = [*LIBRARIES, *BASELINES]
        forecasts, scales = attach_targets(pd.concat(predictions, ignore_index=True), coverage, data, names)
        tables.update(comparison_tables(forecasts, names))
        tables.update(forecasts=forecasts, rmsse_scales=scales, fit_status=pd.DataFrame(fit_records),
                      validation_metrics=pd.DataFrame(validation_scores), permutation_importance=pd.concat(permutation, ignore_index=True),
                      feature_importance=pd.concat(dev_importance, ignore_index=True),
                      training_eligibility=pd.DataFrame(eligibility), unknown_categories=pd.DataFrame(unknowns),
                      training_cohort=data["training_cohort"], evaluation_cohort=data["evaluation_cohort"])
        tables["leakage_checks"] = future_checks(con, data, temporary)
        (temporary / "tables").mkdir()
        for name, frame in tables.items():
            frame.to_parquet(temporary / "tables" / f"{name}.parquet", index=False)
        data["pool"].drop(columns=TARGET).to_parquet(temporary / "historical_features.parquet", index=False)
        data["pool"][KEYS + ["target_date", TARGET]].to_parquet(temporary / "historical_labels.parquet", index=False)
        data["inference"].to_parquet(temporary / "inference_features.parquet", index=False)
        (temporary / "feature_mapping.json").write_text(json.dumps(data["mapping"].as_dict(), indent=2) + "\n", encoding="utf-8")
        if totals(con) != initial or before != {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}:
            raise ValueError("Source fingerprint or warehouse totals changed")
        from .report import figures, write_report
        paths = figures(tables, temporary / "figures")
        train_keys = set(map(tuple, data["training_cohort"][["store_id", "item_id"]].to_numpy()))
        eval_keys = set(map(tuple, data["evaluation_cohort"][["store_id", "item_id"]].to_numpy()))
        summary = {"status": "passed", "config": asdict(config), "selected_rounds": selected_rounds,
                   "feature_columns": data["feature_columns"], "feature_config": asdict(data["feature_config"]),
                   "training_series": config.training_series, "historical_origins": list(data["training_origins"]),
                   "pool_rows": len(data["pool"]), "development_rows": len(data["development_train"]),
                   "validation_rows": len(data["development_validation"]), "validation_end": data["validation_end"],
                   "evaluation_origins": data["origins"], "evaluation_series": len(eval_keys),
                   "evaluation_series_seen_in_training": len(eval_keys & train_keys),
                   "eligible_training": eligibility, "cached_daily_rows": data["cached_daily_rows"],
                   "metrics_common": tables["metrics_common"].to_dict("records"),
                   "validation_metrics": tables["validation_metrics"].to_dict("records"),
                   "fit_status_counts": tables["fit_status"].status.value_counts().to_dict(),
                   "case_status_counts": pd.DataFrame(coverage).status.value_counts().to_dict(),
                   "warnings": int(tables["fit_status"].warning_count.fillna(0).sum()),
                   "clipped_forecast_rows": int(forecasts.raw_prediction.lt(0).sum()),
                   "table_rows": {name: len(frame) for name, frame in tables.items()},
                   "input_fingerprints": before, "source_totals": initial, "inputs_unchanged": True,
                   "runtime_seconds": perf_counter() - started, "dataset_seconds": build_seconds,
                   "resources": {"fit_threads": config.threads, "duckdb_threads": 2,
                                 "duckdb_memory_limit": con.execute("select current_setting('memory_limit')").fetchone()[0],
                                 "pool_dataframe_bytes": int(data["pool"].memory_usage(deep=True).sum())},
                   "versions": {name: version(name) for name in (*LIBRARIES, "scikit-learn", "pandas", "numpy", "duckdb")},
                   "figures": paths, "metric_label": "cohort MAE/WAPE/macro RMSSE; NOT official M5 WRMSSE"}
        write_report(summary, tables, temporary / "experiment_report.md")
        (temporary / "summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        publish(temporary, target)
    return summary, target
