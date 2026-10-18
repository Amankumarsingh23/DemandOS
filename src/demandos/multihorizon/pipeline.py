"""Fit four architectures, freeze validation choices and score identical targets."""

from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from demandos.backtesting.pipeline import publish
from demandos.boosting.config import load_config as load_boosting
from demandos.boosting.dataset import build, eligible_rows, row_digest
from demandos.boosting.models import GlobalModel, TARGET
from demandos.boosting.evaluation import statistical_predictions, attach_targets, comparison_tables, BASELINES
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.features.contracts import KEYS
from demandos.features.pipeline import totals
from demandos.features.validation import generate
from demandos.forecasting.models import forecast
from demandos.forecasting.evaluation import metrics
from demandos.warehouse.cli import settings, connection, build_artifacts
from .models import fit_model, HorizonModels, IndependentMultiOutput
from .vector import complete_rows, target_vectors, VectorLayout
from .recursive import recursive_predict, history_prefix
from .ensemble import select_weight, blend


ARCHITECTURES = ("pooled", "recursive", "direct", "independent_multioutput")


def tagged(frame, architecture, origin):
    result = frame.copy()
    result["model"] = architecture
    result["architecture"] = architecture
    result["model_id"] = f"{architecture}/{pd.Timestamp(origin):%Y%m%d}"
    return result


def seasonal_predictions(inference, daily, feature_config):
    origin = pd.Timestamp(inference.origin_date.iloc[0])
    prefix = history_prefix(daily, inference, origin, feature_config.lookback)
    records = []
    for key, group in prefix.groupby(["store_id", "item_id"], sort=True):
        series = group.sort_values("date").set_index("date").units_sold
        result = forecast(series, 28, "seasonal_naive")
        if result.status != "ok":
            raise ValueError("Required ensemble seasonal-naive component failed")
        for h, (raw, predicted) in enumerate(zip(result.raw_predictions, result.predictions), 1):
            records.append({"store_id": key[0], "item_id": key[1], "origin_date": origin, "horizon": h,
                            "target_date": origin+pd.Timedelta(days=h), "model": "seasonal_naive",
                            "raw_prediction": raw, "prediction": predicted})
    return pd.DataFrame(records)


def predict_architecture(architecture, model, inference, daily, feature_config):
    if architecture == "recursive":
        origin = pd.Timestamp(inference.origin_date.iloc[0])
        prefix = history_prefix(daily, inference, origin, feature_config.lookback)
        result, trace = recursive_predict(model, inference, prefix, feature_config)
        return result, trace
    if architecture == "independent_multioutput":
        return model.predict(model.layout.pack(inference)), None
    return model.predict(inference), None


def fit_fold(data, training, validation, cutoff, directory, config, records, *, phase, rounds=None, probe=None):
    features, fc = data["feature_columns"], data["feature_config"]
    valid_end = data["validation_end"] if validation is not None else None
    current = replace(config, iterations=rounds["pooled"]) if rounds else config
    pooled = fit_model(training, features, current, fc, cutoff, directory/"pooled", records,
                       "pooled", phase, validation=validation, validation_end=valid_end, probe=probe)
    current = replace(config, iterations=rounds["recursive"]) if rounds else config
    one = training[training.horizon.eq(1)].copy()
    val_one = validation[validation.horizon.eq(1)].copy() if validation is not None else None
    recursive = fit_model(one, features, current, fc, cutoff, directory/"recursive", records,
                          "recursive", phase, 1, val_one, valid_end,
                          probe[probe.horizon.eq(1)] if probe is not None else None)
    direct = HorizonModels.fit(training, features, config, fc, cutoff, directory/"direct", records,
                               phase=phase, rounds=rounds["direct"] if rounds else None,
                               validation=validation, validation_end=valid_end, probe=probe)
    full = complete_rows(training, cutoff)
    layout = VectorLayout.from_long(full.drop(columns=TARGET))
    X, Y = layout.pack(full.drop(columns=TARGET)), target_vectors(full)
    vector_validation = (layout.pack(validation.drop(columns=TARGET)), target_vectors(validation)) if validation is not None else None
    vector = IndependentMultiOutput.fit(X, Y, layout, features, config, fc, cutoff,
                directory/"independent_multioutput", records, phase=phase,
                rounds=rounds["independent_multioutput"] if rounds else None,
                validation=vector_validation, validation_end=valid_end,
                probe=layout.pack(probe) if probe is not None else None)
    models = dict(zip(ARCHITECTURES, (pooled, recursive, direct, vector)))
    selected = {"pooled": pooled.rounds, "recursive": recursive.rounds,
                "direct": direct.rounds, "independent_multioutput": vector.estimators.rounds}
    return models, selected, len(X)


def load_fold(directory):
    return {"pooled": GlobalModel.load(directory/"pooled"), "recursive": GlobalModel.load(directory/"recursive"),
            "direct": HorizonModels.load(directory/"direct"),
            "independent_multioutput": IndependentMultiOutput.load(directory/"independent_multioutput")}


def future_check(con, data, origin, inference, daily, directory, expected, weight):
    """Freeze fitted state; regenerate features after changing every future sale/price."""
    day = origin.date().isoformat()
    con.register("feature_keys", data["evaluation_keys"])
    con.execute(f"""create or replace temp view feature_sales as select store_id,item_id,date,
        case when date>date '{day}' then units_sold+1000000 else units_sold end as units_sold from global_daily_panel""")
    con.execute(f"""create or replace temp view feature_prices as select p.* replace(
        case when wm_yr_wk in (select wm_yr_wk from feature_calendar group by wm_yr_wk having min(date)>date '{day}')
        then sell_price*100+999 else sell_price end as sell_price) from main.dim_price_week p""")
    after = generate(con, [origin], data["feature_config"], data["mapping"]).features
    pd.testing.assert_frame_equal(inference.reset_index(drop=True), after.reset_index(drop=True))
    con.execute("create or replace temp view feature_sales as select * from global_daily_panel")
    con.execute("create or replace temp view feature_prices as select * from main.dim_price_week")
    changed = daily.copy()
    changed.loc[pd.to_datetime(changed.date).gt(origin), "units_sold"] += 1000000
    first_prefix = history_prefix(daily, inference, origin, data["feature_config"].lookback)
    second_prefix = history_prefix(changed, after, origin, data["feature_config"].lookback)
    pd.testing.assert_frame_equal(first_prefix, second_prefix)
    pool_changed = data["pool"].copy()
    pool_changed.loc[pool_changed.target_date.gt(origin), TARGET] += 1000000
    allowed = eligible_rows(data["pool"], origin)
    pd.testing.assert_frame_equal(allowed, eligible_rows(pool_changed, origin))
    pd.testing.assert_frame_equal(complete_rows(allowed, origin), complete_rows(eligible_rows(pool_changed, origin), origin))
    records, predictions = [], {}
    for architecture, model in load_fold(directory).items():
        predicted, _ = predict_architecture(architecture, model, after, changed, data["feature_config"])
        predictions[architecture] = predicted
    seasonal = seasonal_predictions(after, changed, data["feature_config"])
    predictions["ensemble"] = blend(predictions["pooled"], seasonal, weight)
    for architecture, predicted in predictions.items():
        original = expected[architecture].sort_values(KEYS).reset_index(drop=True)
        predicted = predicted.sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(original[KEYS+["target_date"]], predicted[KEYS+["target_date"]])
        np.testing.assert_allclose(original.raw_prediction, predicted.raw_prediction, rtol=0, atol=1e-12)
        records.append({"architecture": architecture, "origin": origin, "rows": len(predicted),
                        "future_sales_features_unchanged": True, "future_prices_excluded": True,
                        "future_labels_fit_inputs_unchanged": True, "full_vectors_eligible": True,
                        "frozen_ensemble_weight": weight, "save_load_forecasts_unchanged": True,
                        "fit_digest": row_digest(allowed)})
    return records


def run(root, strategy_config):
    started = perf_counter()
    root, source, warehouse, env = settings(root)
    build_artifacts(root, source, warehouse)
    config = load_boosting(root/strategy_config.boosting_config)
    sources = [source, root/"data/interim/sales_long.parquet", *sorted((root/"data/raw").glob("*.csv"))]
    before = {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}
    target = root/"data/strategies/global_h28"
    with connection(warehouse, env) as con, TemporaryDirectory(prefix=".strategies-", dir=root/"data") as temporary:
        temporary = Path(temporary)
        initial = totals(con)
        expected_totals = json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(v) != str(expected_totals[k]) for k, v in initial.items()):
            raise ValueError("Warehouse baseline totals changed")
        data = build(con, root, config)
        expected_origins = list(pd.to_datetime(["2016-01-31", "2016-02-28", "2016-03-27"]))
        if data["origins"] != expected_origins or 58*(1+len(data["origins"])) > strategy_config.max_native_models:
            raise ValueError("This bounded experiment requires the three declared Season 7 origins and 232 native models")
        daily = con.execute("select store_id,item_id,date,units_sold from global_daily_panel order by store_id,item_id,date").df()
        build_seconds = perf_counter()-started
        records, predictions, coverage, traces, costs, eligibility, leakage = [], [], [], [], [], [], []
        val = data["development_validation"]
        val_features = val.drop(columns=TARGET)
        models, rounds, vector_rows = fit_fold(data, data["development_train"], val,
              pd.Timestamp(config.validation_origin), temporary/"models/development", config, records, phase="development")
        development_predictions, validation_scores = {}, []
        for architecture, model in models.items():
            predicted, _ = predict_architecture(architecture, model, val_features, daily, data["feature_config"])
            development_predictions[architecture] = predicted
            joined = predicted.merge(val[KEYS+["target_date", TARGET]], on=KEYS+["target_date"], validate="one_to_one")
            validation_scores.append({"architecture": architecture, **metrics(joined[TARGET], joined.prediction)})
        seasonal = seasonal_predictions(val_features, daily, data["feature_config"])
        weight, weight_scores = select_weight(development_predictions["pooled"], seasonal, val,
            strategy_config.ensemble_weights, validation_end=data["validation_end"], first_evaluation=data["origins"][0])
        (temporary/"ensemble.json").write_text(json.dumps({"pooled_weight": weight, "seasonal_weight": 1-weight,
             "selection": "minimum internal-validation MAE; ties prefer smaller pooled weight",
             "validation_end": str(data["validation_end"].date()), "candidates": list(strategy_config.ensemble_weights)}, indent=2)+"\n")
        print(f"Development complete: 58 native models; ensemble pooled weight={weight}; complete vectors={vector_rows}", flush=True)
        del models, development_predictions
        for origin in data["origins"]:
            train = eligible_rows(data["pool"], origin)
            inference = data["inference"].loc[data["inference"].origin_date.eq(origin)].reset_index(drop=True)
            directory = temporary/f"models/origin_{origin:%Y%m%d}"
            models, _, vectors = fit_fold(data, train, None, origin, directory, config, records,
                                         phase="evaluation", rounds=rounds, probe=inference)
            eligibility.append({"origin": origin, "pooled_rows": len(train), "one_step_rows": int(train.horizon.eq(1).sum()),
                                "vector_samples": vectors, "vector_long_rows": vectors*28,
                                "target_max": train.target_date.max(), "fit_digest": row_digest(train)})
            fold_predictions = {}
            for architecture, model in models.items():
                start = perf_counter()
                predicted, trace = predict_architecture(architecture, model, inference, daily, data["feature_config"])
                costs.append({"architecture": architecture, "origin": origin, "prediction_seconds": perf_counter()-start,
                              "native_models": 28 if architecture in ("direct", "independent_multioutput") else 1})
                fold_predictions[architecture] = predicted
                if trace is not None:
                    traces.append(trace)
            seasonal = seasonal_predictions(inference, daily, data["feature_config"])
            start = perf_counter()
            fold_predictions["ensemble"] = blend(fold_predictions["pooled"], seasonal, weight)
            costs.append({"architecture": "ensemble", "origin": origin, "prediction_seconds": perf_counter()-start,
                          "native_models": 0, "dependency": "pooled plus seasonal_naive; mixing time only"})
            for architecture, predicted in fold_predictions.items():
                predictions.append(tagged(predicted, architecture, origin))
                coverage.extend({"fit_id": f"{architecture}/{origin:%Y%m%d}", "model": architecture,
                                 "origin": origin, "series_id": row.series_id, "status": "ok", "reason": None}
                                for row in data["evaluation_cohort"].itertuples())
            leakage.extend(future_check(con, data, origin, inference, daily, directory, fold_predictions, weight))
            print(f"Verified {origin:%Y-%m-%d}: {len(train):,} pooled labels, {vectors:,} complete vectors; all strategies cover 28 days", flush=True)
            del models
        baselines, baseline_fits, baseline_coverage = statistical_predictions(data)
        for name, group in baselines.groupby("model"):
            for origin, frame in group.groupby("origin_date"):
                predictions.append(tagged(frame, name, origin))
        coverage.extend(baseline_coverage)
        names = [*ARCHITECTURES, "ensemble", *BASELINES]
        forecasts, scales = attach_targets(pd.concat(predictions, ignore_index=True), coverage, data, names)
        forecasts["origin_date"], forecasts["y_pred"], forecasts["y_true"] = forecasts.origin, forecasts.prediction, forecasts.actual
        forecasts["forecast_status"], forecasts["fit_status"] = forecasts.status, forecasts.status
        tables = comparison_tables(forecasts, names)
        fit_status = pd.DataFrame(records)
        if len(fit_status) != 232 or len(fit_status) > strategy_config.max_native_models or not fit_status.status.eq("ok").all():
            raise ValueError("Native model count/status does not match preregistered complete architecture")
        tables.update(forecasts=forecasts, fit_status=fit_status, baseline_fit_status=pd.DataFrame(baseline_fits),
                      prediction_resources=pd.DataFrame(costs), training_eligibility=pd.DataFrame(eligibility),
                      recursive_trace=pd.concat(traces, ignore_index=True), leakage_checks=pd.DataFrame(leakage),
                      ensemble_validation=weight_scores, validation_metrics=pd.DataFrame(validation_scores),
                      rmsse_scales=scales, training_cohort=data["training_cohort"], evaluation_cohort=data["evaluation_cohort"])
        tables["resources"] = resource_table(fit_status, tables["prediction_resources"])
        previous_path = root/"data/boosting/global_h28/tables/forecasts.parquet"
        reference = {"available": previous_path.is_file()}
        if previous_path.is_file():
            old = pd.read_parquet(previous_path)
            current = forecasts[forecasts.model.eq("pooled")]
            old = old[old.model.eq("lightgbm")]
            joined = current.merge(old, on=["store_id", "item_id", "origin", "horizon", "target_date"], validate="one_to_one", suffixes=("_new", "_old"))
            if len(joined) != len(current) or len(old) != len(current):
                raise ValueError("Season 9 reference keys changed")
            np.testing.assert_array_equal(joined.actual_new, joined.actual_old)
            np.testing.assert_allclose(joined.prediction_new, joined.prediction_old, rtol=1e-10, atol=1e-10)
            reference.update(rows=len(joined), max_prediction_difference=float(np.max(np.abs(joined.prediction_new-joined.prediction_old))))
        (temporary/"tables").mkdir()
        for name, frame in tables.items():
            frame.to_parquet(temporary/"tables"/f"{name}.parquet", index=False)
        if totals(con) != initial or before != {p.relative_to(root).as_posix(): fingerprint(p) for p in sources}:
            raise ValueError("Source input fingerprint or warehouse totals changed")
        summary = {"status": "passed", "strategy_config": asdict(strategy_config), "boosting_config": asdict(config),
                   "feature_columns": data["feature_columns"], "feature_config": asdict(data["feature_config"]),
                   "training_series": len(data["training_cohort"]), "historical_origins": list(data["training_origins"]),
                   "pool_rows": len(data["pool"]), "development_rows": len(data["development_train"]),
                   "development_vector_samples": vector_rows, "validation_rows": len(val),
                   "validation_end": data["validation_end"], "selected_rounds": rounds, "ensemble_pooled_weight": weight,
                   "evaluation_origins": data["origins"], "evaluation_series": len(data["evaluation_cohort"]),
                   "native_models": len(fit_status), "native_fit_status": fit_status.status.value_counts().to_dict(),
                   "baseline_fit_status": tables["baseline_fit_status"].status.value_counts().to_dict(),
                   "metrics_common": tables["metrics_common"].to_dict("records"),
                   "resources": tables["resources"].to_dict("records"), "source_totals": initial,
                   "season9_reference": reference, "table_rows": {k: len(v) for k,v in tables.items()},
                   "input_fingerprints": before, "inputs_unchanged": True,
                   "warnings": int(fit_status.warning_count.sum()), "clipped_forecast_rows": int(forecasts.raw_prediction.lt(0).sum()),
                   "runtime_seconds": perf_counter()-started, "dataset_seconds": build_seconds,
                   "pool_dataframe_bytes": int(data["pool"].memory_usage(deep=True).sum()),
                   "versions": {name: version(name) for name in ("lightgbm", "pandas", "numpy", "duckdb", "pyarrow")}}
        from .report import write_report
        write_report(summary, tables, temporary)
        (temporary/"summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        publish(temporary, target)
    return summary, target


def resource_table(fits, predictions):
    rows = []
    for architecture, group in fits.groupby("architecture", sort=True):
        evaluation = group[group.phase.eq("evaluation")]
        rows.append({"architecture": architecture, "native_models_per_origin": len(evaluation)//3,
                     "total_native_models": len(group), "development_fit_seconds": group.loc[group.phase.eq("development"), "training_seconds"].sum(),
                     "evaluation_fit_seconds": evaluation.training_seconds.sum(),
                     "evaluation_predict_seconds": predictions.loc[predictions.architecture.eq(architecture), "prediction_seconds"].sum(),
                     "all_model_bytes": int(group.model_bytes.sum()), "evaluation_model_bytes": int(evaluation.model_bytes.sum()),
                     "warning_entries": int(group.warning_count.sum())})
    return pd.DataFrame(rows)
