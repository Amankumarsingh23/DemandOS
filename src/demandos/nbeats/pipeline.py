"""Real target-only experiments with checked reference provenance and hard gates."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from demandos.warehouse.cli import settings, build_artifacts, connection
from demandos.features.pipeline import source_views, load_keys, totals
from demandos.timeseries.cohort import training_labels
from demandos.boosting.config import load_config as load_boosting
from demandos.boosting.dataset import select_training_panel, historical_origins
from demandos.backtesting.design import load_config as load_backtesting
from demandos.backtesting.pipeline import extract, publish
from demandos.boosting.evaluation import attach_targets, comparison_tables
from demandos.neural.sequences import Panel, examples
from demandos.neural.pipeline import assert_same_tensors, future_panels, prediction_rows
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from .dataset import TargetDataset, SCALING
from .training import train, save, load, predict


VARIANTS = ("generic", "interpretable")
REFERENCES = ("gru", "lstm", "lightgbm", "tsb", "seasonal_naive")
MODELS = ("nbeats_generic", "nbeats_interpretable", *REFERENCES)


def validate_reference_cutoff(payload, origin):
    """Neural metadata is flat; native GBDT payloads nest their fit metadata."""
    meta = payload.get("metadata", payload)
    target = pd.Timestamp(meta.get("fit_target_max", meta.get("training_target_max")))
    cutoff = pd.Timestamp(meta.get("fit_cutoff", meta.get("allowed_cutoff")))
    if pd.isna(target) or pd.isna(cutoff) or target > origin or cutoff > origin:
        raise ValueError("Missing or future-fitted cached reference model cutoffs")


def prepare(con, root, config):
    source_views(con)
    frames, cohort, schedule, _, _, _ = extract(con, root, load_backtesting(root/"configs/backtesting.toml"))
    evaluation = cohort[cohort.kind.eq("store_item")].copy()
    origins = list(pd.DatetimeIndex(schedule.origin.unique()).sort_values())
    boost = load_boosting(root/"configs/boosting.toml")
    start = pd.Timestamp(con.execute("select min(date) from main.dim_calendar").fetchone()[0])
    labels, _, _ = training_labels(con, root, start, pd.Timestamp(boost.selection_cutoff))
    selected = select_training_panel(labels, boost)
    train_keys = load_keys(con, boost.selection_cutoff, selected)
    eval_keys = load_keys(con, origins[0], evaluation)
    metadata = pd.concat([train_keys, eval_keys]).drop_duplicates(["store_id", "item_id"])
    train_origins = historical_origins(boost, origins[-1])
    con.register("nbeats_keys", metadata[["store_id", "item_id"]])
    daily = con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price
        from main.fact_daily_sales f join nbeats_keys k using(store_id,item_id)
        where date between ? and ? order by store_id,item_id,date""",
        [(train_origins[0]-pd.Timedelta(days=config.lookback-1)).date(), (origins[-1]+pd.Timedelta(days=config.horizon)).date()]).fetchdf()
    calendar = con.execute("select * from main.dim_calendar order by date").fetchdf()
    return {"panel": Panel(daily, calendar, metadata), "daily": daily, "calendar": calendar, "metadata": metadata,
            "origins": origins, "training_origins": train_origins, "training_cohort": selected, "train_keys": train_keys,
            "evaluation_cohort": evaluation, "eval_keys": eval_keys, "frames": frames, "boost": boost}


def references(root, sources, data):
    directory = root/"data/neural/global_h28"
    if not (directory/"summary.json").is_file():
        raise FileNotFoundError("Run python -m demandos.neural --root . first: verified Season 12 references are required")
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "passed" or summary["source_fingerprints"] != sources or summary["boosting_config"] != asdict(data["boost"]):
        raise ValueError("Cached Season 12 source/configuration provenance mismatch; rerun Season 12")
    columns = ["store_id", "item_id"]
    training = pd.read_parquet(directory/"tables/training_cohort.parquet")
    for a, b in [(training, data["training_cohort"]),
                 (pd.read_parquet(directory/"tables/evaluation_cohort.parquet"), data["evaluation_cohort"])]:
        pd.testing.assert_frame_equal(a[columns].sort_values(columns).reset_index(drop=True), b[columns].sort_values(columns).reset_index(drop=True))
    if list(pd.to_datetime(summary["evaluation_origins"])) != data["origins"]:
        raise ValueError("Reference evaluation origins differ")
    frame = pd.read_parquet(directory/"tables/forecasts.parquet")
    if set(frame.model) != set(REFERENCES) or not frame.status.eq("ok").all() or frame.groupby("model").size().ne(1092).any():
        raise ValueError("Reference coverage is incomplete")
    metric_columns = ["model","mae","wape","macro_rmsse","observations","rmsse_cases","undefined_rmsse_cases"]
    actual_metrics = comparison_tables(frame,REFERENCES)["metrics_common"][metric_columns].sort_values("model").reset_index(drop=True)
    expected_metrics = pd.DataFrame(summary["metrics_common"])[metric_columns].sort_values("model").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_metrics,expected_metrics,check_dtype=False,rtol=1e-12,atol=1e-12)
    for origin in data["origins"]:
        for model in ("gru", "lstm", "lightgbm"):
            meta = json.loads((directory/"models"/model/f"origin_{origin:%Y%m%d}"/"metadata.json").read_text())
            validate_reference_cutoff(meta, origin)
    values = frame[["store_id", "item_id", "origin", "target_date", "horizon", "model", "prediction", "raw_prediction"]].rename(columns={"origin": "origin_date"})
    coverage = frame[["series_id", "origin", "model", "status", "reason"]].drop_duplicates().to_dict("records")
    return values, coverage, frame, {"forecasts": fingerprint(directory/"tables/forecasts.parquet"),
        "summary": fingerprint(directory/"summary.json"), "mode": "source/config/cohort-bound verified Season 12 predictions; not retrained",
        "timings": pd.read_parquet(directory/"tables/timings.parquet").to_dict("records")}


def forecast_tables(rows, raw, scored, parts, model):
    values = prediction_rows(rows, scored, f"nbeats_{model}")
    values["raw_prediction"] = raw.reshape(-1)
    blocks = []
    for index, row in enumerate(rows.itertuples()):
        for block in range(parts.shape[1]):
            for h in range(1, 29):
                blocks.append({"store_id": row.store_id, "item_id": row.item_id, "origin_date": row.origin_date,
                               "target_date": row.origin_date+pd.Timedelta(days=h), "horizon": h, "model": f"nbeats_{model}",
                               "block": block, "contribution": parts[index, block, h-1]})
    return values, pd.DataFrame(blocks)


def run(root, config):
    began = perf_counter()
    root, source, warehouse, env = settings(root)
    build_artifacts(root, source, warehouse)
    paths = [source, root/"data/interim/sales_long.parquet", *sorted((root/"data/raw").glob("*.csv"))]
    before = {p.relative_to(root).as_posix(): fingerprint(p) for p in paths}
    output = root/"data/nbeats/global_h28"
    with connection(warehouse, env) as con, TemporaryDirectory(prefix=".nbeats-", dir=root/"data") as temporary:
        temporary = Path(temporary)
        initial = totals(con)
        expected = json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(value) != str(expected[name]) for name, value in initial.items()):
            raise ValueError("Verified warehouse baseline changed")
        data = prepare(con, root, config)
        reference, coverage, reference_scored, provenance = references(root, before, data)
        cutoff = pd.Timestamp(data["boost"].validation_origin)
        validation_end = cutoff+pd.Timedelta(days=config.horizon)
        if validation_end >= data["origins"][0]:
            raise ValueError("Development validation overlaps the evaluation origins")
        rows = examples(data["train_keys"], data["training_origins"], config, cutoff)
        development = TargetDataset(data["panel"], rows, config, label_cutoff=cutoff)
        val_rows = examples(data["train_keys"], [cutoff], config)
        validation = TargetDataset(data["panel"], val_rows, config, label_cutoff=validation_end)
        probe = TargetDataset(data["panel"], val_rows, config)
        records, curves, selected, predictions, blocks, checks, eligibility, audits, history = [], [], {}, [reference], [], [], [], [], []
        print(f"N-BEATS: {len(data['train_keys'])} training series; {len(development)} development + {len(validation)} validation windows; {len(data['daily']):,} daily rows cached", flush=True)
        for variant in VARIANTS:
            model, meta = train(variant, development, config, cutoff=cutoff, validation=validation, validation_cutoff=validation_end)
            path = temporary/"models"/variant/"development"
            save(model, meta, path)
            loaded, _ = load(path, device=config.device)
            for a, b in zip(predict(model, probe, config), predict(loaded, probe, config)):
                np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
            selected[variant] = meta["selected_epochs"]
            records.append({**meta, "phase": "development", "model_bytes": (path/"weights.pt").stat().st_size})
        print(f"Frozen December-selected epoch counts: {selected}", flush=True)
        for origin in [cutoff, *data["origins"]]:
            fitting_rows = examples(data["train_keys"], data["training_origins"], config, origin)
            eligibility.append({"fit_cutoff": origin, "candidate_windows": len(data["training_origins"])*len(data["train_keys"]),
                                "eligible_windows": len(fitting_rows), "label_values": len(fitting_rows)*config.horizon,
                                "excluded_incomplete_vectors": len(data["training_origins"])*len(data["train_keys"])-len(fitting_rows),
                                "reason": "target_end_after_fit_cutoff", "target_max": fitting_rows.origin_date.max()+pd.Timedelta(days=config.horizon)})
            if origin == cutoff:
                continue
            fitting = TargetDataset(data["panel"], fitting_rows, config, label_cutoff=origin)
            infer_rows = examples(data["eval_keys"], [origin], config)
            inference = TargetDataset(data["panel"], infer_rows, config)
            altered = [TargetDataset(p, infer_rows, config) for p in future_panels(data["daily"], data["calendar"], data["metadata"], origin)]
            for alternative in altered:
                assert_same_tensors(inference, alternative)
            for panel in future_panels(data["daily"], data["calendar"], data["metadata"], origin):
                assert_same_tensors(fitting, TargetDataset(panel, fitting_rows, config, label_cutoff=origin))
            audits.extend(inference.audits)
            for variant in VARIANTS:
                model, meta = train(variant, fitting, config, cutoff=origin, epochs=selected[variant])
                path = temporary/"models"/variant/f"origin_{origin:%Y%m%d}"
                save(model, meta, path)
                start = perf_counter()
                raw, scored, parts = predict(model, inference, config)
                meta["prediction_seconds"] = perf_counter()-start
                loaded, _ = load(path, device=config.device)
                comparisons = [predict(loaded, inference, config), *[predict(loaded, a, config) for a in altered]]
                for result in comparisons:
                    for a, b in zip((raw, scored, parts), result):
                        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
                checks.append({"model": f"nbeats_{variant}", "origin": origin, "rows": raw.size, "tensors_exact": True,
                               "future_sales_mutated_and_removed": True, "future_prices_excluded": True, "future_labels_excluded": True,
                               "eligible_training_tensors_unchanged": True, "save_load_equal": True,
                               "max_prediction_difference": max(float(np.abs(raw-r[0]).max()) for r in comparisons),
                               "component_max_difference": float(np.abs(raw-parts.sum(axis=1)).max()), "atol": 1e-6, "rtol": 1e-6})
                values, contributions = forecast_tables(infer_rows, raw, scored, parts, variant)
                contributions["component"] = [model.blocks[i].kind for i in contributions.block]
                predictions.append(values); blocks.append(contributions)
                records.append({**meta, "phase": "evaluation_refit", "model_bytes": (path/"weights.pt").stat().st_size})
                for member in data["evaluation_cohort"].itertuples():
                    coverage.append({"model": f"nbeats_{variant}", "origin": origin, "series_id": member.series_id,
                                     "status": "ok", "reason": None})
            # Actual historical prefixes used for representative plots, never future predictors.
            for row, x, scale in zip(infer_rows.itertuples(), inference.tensors[0].numpy(), inference.tensors[2].numpy()):
                history.extend({"store_id": row.store_id, "item_id": row.item_id, "origin": origin,
                                "date": day, "observed_units": float(value*scale)}
                               for day, value in zip(pd.date_range(origin-pd.Timedelta(days=55), origin), x))
            print(f"Verified {origin:%Y-%m-%d}: {len(fitting)} complete vectors, both variants future-invariant", flush=True)
        forecasts, scales = attach_targets(pd.concat(predictions, ignore_index=True), coverage, data, MODELS)
        # Reattach real actuals/scales to cached predictions and verify the earlier scoring state.
        old = forecasts[forecasts.model.isin(REFERENCES)]
        match = old.merge(reference_scored, on=["store_id","item_id","origin","horizon","target_date","model"], validate="one_to_one", suffixes=("_new","_old"))
        for name in ("actual", "scale", "prediction", "raw_prediction"):
            np.testing.assert_allclose(match[f"{name}_new"].astype(float), match[f"{name}_old"].astype(float), equal_nan=True, rtol=1e-12, atol=1e-12)
        tables = comparison_tables(forecasts, MODELS)
        if len(tables["comparison_membership"]) != 1092 or len(forecasts) != 7644:
            raise ValueError("Matched forecast keys/coverage changed")
        block_frame = pd.concat(blocks, ignore_index=True)
        point = ["store_id","item_id","origin_date","target_date","horizon","model"]
        component = block_frame.groupby(point+["component"]).contribution.sum().unstack("component").reset_index()
        component = component.merge(pd.concat(predictions[1:],ignore_index=True), on=point, validate="one_to_one")
        component["component_total"] = component[["generic","trend","seasonality"]].fillna(0).sum(axis=1)
        np.testing.assert_allclose(component.component_total, component.raw_prediction, atol=1e-4, rtol=1e-6)
        component["clipping_adjustment"] = component.prediction-component.raw_prediction
        for record in records:
            curves.extend({**row,"variant":record["variant"],"phase":record["phase"],"fit_cutoff":record["fit_cutoff"]} for row in record["curves"])
        fit_records = [{k:v for k,v in r.items() if k not in ("curves","config","scaling")} for r in records]
        for record in fit_records:
            for key,value in list(record.items()):
                if isinstance(value, (list,dict)): record[key] = json.dumps(value)
        tables.update(forecasts=forecasts, scales=scales, block_forecasts=block_frame, components=component,
                      training_curves=pd.DataFrame(curves), fit_status=pd.DataFrame(fit_records), eligibility=pd.DataFrame(eligibility),
                      tensor_audits=pd.DataFrame(audits), leakage_checks=pd.DataFrame(checks), history=pd.DataFrame(history),
                      training_cohort=data["training_cohort"], evaluation_cohort=data["evaluation_cohort"],
                      reference_timings=pd.DataFrame(provenance["timings"]))
        if totals(con) != initial or before != {p.relative_to(root).as_posix(): fingerprint(p) for p in paths}:
            raise ValueError("Raw/processed inputs or warehouse totals changed")
        (temporary/"tables").mkdir()
        for name, frame in tables.items(): frame.to_parquet(temporary/"tables"/f"{name}.parquet",index=False)
        summary = {"status":"passed", "config":asdict(config), "scaling":SCALING,
                   "training_series":len(data["train_keys"]), "daily_cache_rows":len(data["daily"]),
                   "development_windows":len(development), "validation_windows":len(validation),
                   "development_cutoff":cutoff,"validation_end":validation_end,"evaluation_origins":data["origins"],
                   "selected_epochs":selected,"fit_records":records,"eligibility":eligibility,"leakage_checks":checks,
                   "metrics_common":tables["metrics_common"].to_dict("records"),"reference_provenance":provenance,
                   "source_fingerprints":before,"inputs_unchanged":True,"warehouse_totals":initial,
                   "table_rows":{k:len(v) for k,v in tables.items()},"runtime_seconds":perf_counter()-began,
                   "versions":{name:version(name) for name in ("torch","numpy","pandas","duckdb")},
                   "resources":{"device":config.device,"threads":config.threads,"workers":0,"mixed_precision":False,
                                "largest_tensor_cache_bytes":sum(t.numel()*t.element_size() for t in fitting.tensors),
                                "peak_process_memory":"Not sampled; bounded tensor bytes reported"},
                   "limitations":["Retrospective 13-series benchmark, not prospective testing or official M5 WRMSSE",
                                  "Target-only N-BEATS versus covariate-enriched GRU/LSTM/LightGBM; not architecture-only isolation",
                                  "Model-attributed components are not identified economic or causal components",
                                  "Recorded sales are not verified unconstrained demand; no inventory-savings claim",
                                  "Cold/short-history masks supported, but cold-start accuracy not established"]}
        from .report import artifacts
        artifacts(summary,tables,temporary)
        (temporary/"summary.json").write_text(json.dumps(json_ready(summary),default=str,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (temporary/"leakage_verification.json").write_text(json.dumps(json_ready(checks),default=str,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        publish(temporary,output)
    return summary,output
