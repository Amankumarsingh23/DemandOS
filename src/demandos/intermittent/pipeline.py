"""Bounded real M5 fitting, pre-evaluation policy selection and explicit cold stress."""
from dataclasses import asdict,replace
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from demandos.boosting.config import load_config as load_boosting
from demandos.boosting.dataset import build,eligible_rows,row_digest
from demandos.boosting.models import GlobalModel,TARGET
from demandos.backtesting.pipeline import publish
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.features.pipeline import totals,load_keys
from demandos.forecasting.models import forecast
from demandos.multihorizon.models import fit_model
from demandos.warehouse.cli import settings,connection,build_artifacts
from .hurdle import HurdleModel
from .dataset import stress_cohort,observed_frames,origin_features
from .diagnostics import prefix
from .priors import PriorState
from .policy import SegmentPolicy
from .evaluation import POINT,BASELINES,MODELS,attach_actuals,tables as score_tables


def fit_fold(data,training,validation,origin,directory,config,records,rounds=None,probe=None):
    directory.mkdir(parents=True,exist_ok=True)
    pooled_config = replace(config,iterations=rounds["pooled"]) if rounds else config
    pooled = fit_model(training,data["feature_columns"],pooled_config,data["feature_config"],origin,directory/"pooled",records,
        "pooled","development" if validation is not None else "evaluation",validation=validation,
        validation_end=data["validation_end"] if validation is not None else None,probe=probe)
    start = perf_counter()
    hurdle = HurdleModel(data["feature_columns"],config,data["feature_config"]).fit(training,allowed_cutoff=origin,
        validation=validation,validation_cutoff=data["validation_end"] if validation is not None else None,
        rounds=rounds["hurdle"] if rounds else None)
    fit_seconds = perf_counter()-start
    hurdle.save(directory/"hurdle")
    sample = validation.drop(columns=TARGET) if validation is not None else probe
    before = hurdle.predict(sample)
    after = HurdleModel.load(directory/"hurdle").predict(sample)
    np.testing.assert_allclose(before[["p_positive","conditional_positive_mean","prediction"]],
                               after[["p_positive","conditional_positive_mean","prediction"]],rtol=1e-10,atol=1e-10)
    metadata = [m.metadata for m in (hurdle.occurrence,hurdle.size) if m is not None]
    records.append({"architecture":"hurdle","phase":"development" if validation is not None else "evaluation",
                    "cutoff":origin,"rows":len(training),"positive_rows":hurdle.state["positive_rows"],
                    "status":"ok","training_seconds":fit_seconds,"fit_digest":row_digest(training),
                    "target_max":training.target_date.max(),"warning_count":sum(len(m["warnings"])+len(m["native_warning_lines"]) for m in metadata),
                    "save_load_equal":True,"model_bytes":sum(p.stat().st_size for p in (directory/"hurdle").rglob("model.txt")),
                    "native_models":len(metadata),"hurdle_rounds":json.dumps(hurdle.state["rounds"])})
    return pooled,hurdle,{"pooled":pooled.rounds,"hurdle":hurdle.state["rounds"]}


def forecast_origin(cohort,origin,features,diagnostics,frames,prior,pooled,hurdle,config):
    result,prior_rows,baseline_records = [],[],[]
    ml = {}
    if not features.empty:
        ml = {"pooled":pooled.predict(features),"hurdle":hurdle.predict(features)}
    for row in cohort.itertuples():
        history = frames[(row.store_id,row.item_id)]
        limit = int(row.visible_days) if row.visible_days>=0 else None
        observed = prefix(history,origin,limit)
        category = None if row.scenario=="unknown_metadata_0" else row.cat_id
        estimated = prior.estimate(row.store_id,row.item_id,category,observed,origin)
        prior_rows.append({"store_id":row.store_id,"item_id":row.item_id,"origin_date":origin,**estimated})
        for model in ("pooled","hurdle",*BASELINES,"cold_prior"):
            selected = None
            status,reason = "ok",None
            raw,predicted = None,None
            if model in ("pooled","hurdle"):
                selected = ml[model][ml[model].store_id.eq(row.store_id)&ml[model].item_id.eq(row.item_id)].sort_values("horizon") if model in ml else pd.DataFrame()
                if len(selected)==28:
                    raw,predicted = selected.raw_prediction.to_numpy(float),selected.prediction.to_numpy(float)
                else:
                    status,reason = "skipped","no_observed_history_outside_feature_contract"
            elif model=="cold_prior":
                if estimated["prior_prediction"] is None:
                    status,reason = "skipped",estimated["prior_status"]
                else:
                    raw = predicted = np.repeat(estimated["prior_prediction"],28)
                    reason = estimated["prior_status"]
            else:
                start = perf_counter()
                if observed.empty:
                    status,reason = "skipped","no_observed_history"
                    warnings=[]
                else:
                    series = pd.Series(observed.units_sold.to_numpy(),index=pd.DatetimeIndex(observed.date))
                    local = forecast(series,28,model)
                    status,reason,raw,predicted = local.status,local.reason,local.raw_predictions,local.predictions
                    warnings=local.warnings
                baseline_records.append({"model":model,"store_id":row.store_id,"item_id":row.item_id,"origin_date":origin,
                    "status":status,"reason":reason,"runtime_seconds":perf_counter()-start,"warnings":json.dumps(warnings)})
            for h in range(1,29):
                point = {"store_id":row.store_id,"item_id":row.item_id,"origin_date":origin,"horizon":h,
                         "target_date":origin+pd.Timedelta(days=h),"model":model,"status":status,"reason":reason,
                         "raw_prediction":raw[h-1] if status=="ok" else np.nan,"prediction":predicted[h-1] if status=="ok" else np.nan}
                if model=="hurdle" and status=="ok":
                    for column in ("p_positive","raw_positive_size","conditional_positive_mean","hurdle_status"):
                        point[column] = selected.iloc[h-1][column]
                result.append(point)
    result = pd.DataFrame(result).merge(diagnostics[["store_id","item_id","origin_date","regime","cold_status","visible_history_days"]],
        on=["store_id","item_id","origin_date"],validate="many_to_one")
    return result,pd.DataFrame(prior_rows),pd.DataFrame(baseline_records)


def invariance(con,cohort,origin,data,frames,config,expected_features,expected_diagnostics,expected_prior,
               expected_forecasts,policy,directory):
    changed = {}
    for key,frame in frames.items():
        frame = frame.copy()
        frame.loc[pd.to_datetime(frame.date).gt(origin),"units_sold"] += 1000000
        changed[key] = frame
    day = origin.date().isoformat()
    con.execute(f"""create or replace temp view sparse_changed as select * replace(
        case when date>date '{day}' then units_sold+1000000 else units_sold end as units_sold) from sparse_daily""")
    con.execute(f"""create or replace temp view feature_prices as select * replace(
        case when wm_yr_wk in (select wm_yr_wk from feature_calendar group by wm_yr_wk having min(date)>date '{day}')
        then sell_price+99999 else sell_price end as sell_price) from main.dim_price_week""")
    features,diagnostics = origin_features(con,cohort,origin,data,changed,config,sales_view="sparse_changed")
    pd.testing.assert_frame_equal(expected_features,features)
    pd.testing.assert_frame_equal(expected_diagnostics,diagnostics)
    con.execute(f"""create or replace temp view sparse_prior_source as select * replace(
        case when date>date '{day}' then units_sold+1000000 else units_sold end as units_sold) from main.fact_daily_sales""")
    prior = PriorState.from_warehouse(con,origin,config,relation="sparse_prior_source")
    pooled = GlobalModel.load(directory/"pooled")
    hurdle = HurdleModel.load(directory/"hurdle")
    forecasts,prior_rows,_ = forecast_origin(cohort,origin,features,diagnostics,changed,prior,pooled,hurdle,config)
    pd.testing.assert_frame_equal(expected_prior,prior_rows)
    routed,_ = policy.predict(forecasts,origin)
    forecasts = combine(forecasts,routed,diagnostics)
    sort = POINT+["model"]
    before,after = expected_forecasts.sort_values(sort).reset_index(drop=True),forecasts.sort_values(sort).reset_index(drop=True)
    pd.testing.assert_frame_equal(before[sort+["status","regime"]],after[sort+["status","regime"]])
    np.testing.assert_allclose(before.prediction,after.prediction,rtol=0,atol=1e-12,equal_nan=True)
    for column in ("p_positive","conditional_positive_mean"):
        np.testing.assert_allclose(before[column],after[column],rtol=0,atol=1e-12,equal_nan=True)
    altered = data["pool"].copy()
    altered.loc[altered.target_date.gt(origin),TARGET] += 1000000
    pd.testing.assert_frame_equal(eligible_rows(data["pool"],origin),eligible_rows(altered,origin))
    con.execute("create or replace temp view feature_prices as select * from main.dim_price_week")
    con.execute("create or replace temp view feature_sales as select * from sparse_daily")
    return {"origin":origin,"features_unchanged":True,"diagnostics_unchanged":True,"priors_unchanged":True,
            "frozen_predictions_routes_weights_unchanged":True,"training_label_eligibility_unchanged":True,
            "future_price_exclusion":True,"forecast_rows":len(after)}


def combine(base,routed,diagnostics):
    routed = routed.merge(diagnostics[["store_id","item_id","origin_date","regime","cold_status","visible_history_days"]],
        on=["store_id","item_id","origin_date"],validate="many_to_one")
    return pd.concat([base,routed],ignore_index=True)


def run(root,config):
    started = perf_counter()
    root,source,warehouse,env = settings(root)
    build_artifacts(root,source,warehouse)
    boost = load_boosting(root/config.boosting_config)
    paths = [source,root/"data/interim/sales_long.parquet",*sorted((root/"data/raw").glob("*.csv"))]
    before = {p.relative_to(root).as_posix():fingerprint(p) for p in paths}
    target = root/"data/intermittent/global_h28"
    with connection(warehouse,env) as con,TemporaryDirectory(prefix=".intermittent-",dir=root/"data") as temporary:
        temporary = Path(temporary)
        initial = totals(con)
        expected = json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(v)!=str(expected[k]) for k,v in initial.items()):
            raise ValueError("Warehouse baseline totals changed")
        data = build(con,root,boost)
        print(f"Historical pool ready: {len(data['pool']):,} labels",flush=True)
        if [str(t.date()) for t in data["origins"]] != ["2016-01-31","2016-02-28","2016-03-27"]:
            raise ValueError("Season 11 uses the declared three historical benchmark origins")
        cohort = stress_cohort(con,data,config)
        daily,frames = observed_frames(con,data,cohort)
        print(f"Selected {len(cohort)} distinct evaluation series; cached {len(daily):,} real daily rows",flush=True)
        records,all_predictions,all_diagnostics,all_priors,all_local,all_decisions,checks,timings = [],[],[],[],[],[],[],[]
        validation_origin = pd.Timestamp(boost.validation_origin)
        train_meta = load_keys(con,validation_origin,data["training_cohort"])
        train_meta = train_meta.assign(cohort="development",scenario="observed",visible_days=-1,series_id=lambda f:"store_item/"+f.store_id+"/"+f.item_id)
        val_features,val_diagnostics = origin_features(con,train_meta,validation_origin,data,frames,config)
        # Model early stopping and policy evidence use the same strictly earlier validation block.
        pooled,hurdle,rounds = fit_fold(data,data["development_train"],data["development_validation"],validation_origin,
                                       temporary/"models/development",boost,records)
        val_prior = PriorState.from_warehouse(con,validation_origin,config)
        validation,validation_priors,validation_local = forecast_origin(train_meta,validation_origin,val_features,val_diagnostics,frames,val_prior,pooled,hurdle,config)
        labels = data["development_validation"][["store_id","item_id","origin_date","horizon","target_date",TARGET]]
        validation = validation.merge(labels,on=POINT,validate="many_to_one").rename(columns={TARGET:"actual"})
        policy = SegmentPolicy(config).fit(validation,validation_end=data["validation_end"],first_evaluation=data["origins"][0])
        policy.save(temporary/"policy.json")
        policy = SegmentPolicy.load(temporary/"policy.json")
        print(f"Developed hurdle/pooled models; policy frozen through {data['validation_end']:%Y-%m-%d}; evaluation cohorts {cohort.groupby('cohort').size().to_dict()}",flush=True)
        for origin in data["origins"]:
            training = eligible_rows(data["pool"],origin)
            features,diagnostics = origin_features(con,cohort,origin,data,frames,config)
            directory = temporary/f"models/origin_{origin:%Y%m%d}"
            pooled,hurdle,_ = fit_fold(data,training,None,origin,directory,boost,records,rounds,features)
            start = perf_counter()
            prior = PriorState.from_warehouse(con,origin,config)
            base,prior_rows,local = forecast_origin(cohort,origin,features,diagnostics,frames,prior,pooled,hurdle,config)
            routed,decisions = policy.predict(base,origin)
            complete = combine(base,routed,diagnostics)
            timings.append({"origin":origin,"prediction_prior_policy_seconds":perf_counter()-start,"feature_rows":len(features)})
            checks.append(invariance(con,cohort,origin,data,frames,config,features,diagnostics,prior_rows,complete,policy,directory))
            all_predictions.append(complete);all_diagnostics.append(diagnostics);all_priors.append(prior_rows)
            all_local.append(local);all_decisions.append(decisions)
            print(f"Verified {origin:%Y-%m-%d}: {len(training):,} fitting rows; {len(complete):,} planned prediction rows",flush=True)
        forecasts = attach_actuals(pd.concat(all_predictions,ignore_index=True),cohort,frames,data["origins"])
        tables = score_tables(forecasts)
        tables.update(forecasts=forecasts,cohort=cohort,diagnostics=pd.concat(all_diagnostics,ignore_index=True),
            cold_priors=pd.concat(all_priors,ignore_index=True),fit_status=pd.DataFrame(records),baseline_fit_status=pd.concat(all_local,ignore_index=True),
            policy_decisions=pd.concat(all_decisions,ignore_index=True),validation_forecasts=validation,validation_diagnostics=val_diagnostics,
            validation_priors=validation_priors,leakage_checks=pd.DataFrame(checks),timings=pd.DataFrame(timings))
        previous = root/"data/boosting/global_h28/tables/forecasts.parquet"
        reference = {"available":previous.is_file()}
        if previous.is_file():
            old = pd.read_parquet(previous)
            new = forecasts[forecasts.cohort.eq("matched")&forecasts.model.eq("pooled")]
            old = old[old.model.eq("lightgbm")]
            joined = new.merge(old,on=["store_id","item_id","origin","horizon","target_date"],validate="one_to_one",suffixes=("_new","_old"))
            if len(joined)!=1092:
                raise ValueError("Continuity comparison keys changed")
            np.testing.assert_allclose(joined.prediction_new,joined.prediction_old,rtol=1e-10,atol=1e-10)
            reference.update(rows=len(joined),max_prediction_difference=float(abs(joined.prediction_new-joined.prediction_old).max()))
        if totals(con)!=initial or before!={p.relative_to(root).as_posix():fingerprint(p) for p in paths}:
            raise ValueError("Original sources or warehouse totals changed")
        (temporary/"tables").mkdir()
        for name,frame in tables.items():
            frame.to_parquet(temporary/"tables"/f"{name}.parquet",index=False)
        summary = {"status":"passed","config":asdict(config),"boosting_config":asdict(boost),"feature_config":asdict(data["feature_config"]),
            "selected_rounds":rounds,"training_series":len(data["training_cohort"]),"pool_rows":len(data["pool"]),
            "historical_origins":list(data["training_origins"]),"development_rows":len(data["development_train"]),
            "validation_rows":len(data["development_validation"]),"validation_end":data["validation_end"],"evaluation_origins":data["origins"],
            "cohort_series":cohort.groupby("cohort").size().to_dict(),"cohort_selection_counts":cohort.groupby(["cohort","selection_reason"]).size().to_dict(),
            "cold_items_excluded_from_global_training":not set(cohort[cohort.cohort.eq("cold_simulation")].item_id)&set(data["pool"].item_id),
            "cached_daily_rows":len(daily),"metrics_common":tables["metrics_common"].to_dict("records"),
            "occurrence_metrics":tables["occurrence_metrics"].to_dict("records"),"positive_size_metrics":tables["positive_size_metrics"].to_dict("records"),
            "fit_status":tables["fit_status"].status.value_counts().to_dict(),"baseline_fit_status":tables["baseline_fit_status"].status.value_counts().to_dict(),
            "forecast_status":forecasts.status.value_counts().to_dict(),"native_models":sum(1 for p in (temporary/"models").rglob("model.txt")),
            "native_model_bytes":sum(p.stat().st_size for p in (temporary/"models").rglob("model.txt")),
            "warnings":int(tables["fit_status"].warning_count.sum()),"table_rows":{k:len(v) for k,v in tables.items()},
            "policy":policy.state,"input_fingerprints":before,"inputs_unchanged":True,"source_totals":initial,"season9_reference":reference,
            "runtime_seconds":perf_counter()-started,"optional_zip":"not implemented; no stockout or latent-demand labels estimated",
            "versions":{name:version(name) for name in ("lightgbm","pandas","numpy","duckdb","pyarrow")}}
        # JSON object keys must be scalar strings, never tuple-valued pandas group keys.
        summary["cohort_selection_counts"] = {"/".join(key):value for key,value in summary["cohort_selection_counts"].items()}
        from .report import write_report
        write_report(summary,tables,temporary)
        (temporary/"summary.json").write_text(json.dumps(json_ready(summary),default=str,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        publish(temporary,target)
    return summary,target
