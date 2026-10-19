"""Chronological real M5 TFT experiment composed with existing warehouse/metrics."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from demandos.warehouse.cli import settings,connection,build_artifacts
from demandos.features.pipeline import totals
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.backtesting.pipeline import publish
from demandos.backtesting.metrics import paired_losses
from demandos.boosting.evaluation import attach_targets,comparison_tables
from demandos.nbeats.pipeline import prepare,references as season12_references,validate_reference_cutoff
from demandos.nbeats.config import load_config as load_nbeats
from demandos.neural.sequences import Preprocessor,SequenceDataset,examples
from demandos.neural.pipeline import future_panels,assert_same_tensors,prediction_rows
from .training import train,predict,save,load
from .interpretation import tables as interpretation_tables


BASELINES=("nbeats_generic","nbeats_interpretable","gru","lstm","lightgbm","seasonal_naive","tsb")
MODELS=("tft",*BASELINES)


def reference_forecasts(root,sources,data):
    season12_references(root,sources,data)
    directory=root/"data/nbeats/global_h28"
    if not (directory/"summary.json").is_file(): raise FileNotFoundError("Run python -m demandos.nbeats first: verified Season 13 references are required")
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    if summary.get("status")!="passed" or summary["source_fingerprints"]!=sources or summary["config"]!=asdict(load_nbeats(root/"configs/nbeats.toml")):
        raise ValueError("Season 13 source/config provenance changed")
    for name,current in (("training_cohort",data["training_cohort"]),("evaluation_cohort",data["evaluation_cohort"])):
        columns=["store_id","item_id"]
        saved=pd.read_parquet(directory/"tables"/f"{name}.parquet")
        pd.testing.assert_frame_equal(saved[columns].sort_values(columns).reset_index(drop=True),current[columns].sort_values(columns).reset_index(drop=True))
    for origin in data["origins"]:
        for variant in ("generic","interpretable"):
            payload=json.loads((directory/"models"/variant/f"origin_{origin:%Y%m%d}"/"metadata.json").read_text())
            validate_reference_cutoff(payload,origin)
    frame=pd.read_parquet(directory/"tables/forecasts.parquet")
    if set(frame.model)!=set(BASELINES) or not frame.status.eq("ok").all() or frame.groupby("model").size().ne(1092).any():
        raise ValueError("Incomplete cached reference population")
    columns=["model","mae","wape","macro_rmsse","observations","rmsse_cases","undefined_rmsse_cases"]
    computed=comparison_tables(frame,BASELINES)["metrics_common"][columns].sort_values("model").reset_index(drop=True)
    expected=pd.DataFrame(summary["metrics_common"])[columns].sort_values("model").reset_index(drop=True)
    pd.testing.assert_frame_equal(computed,expected,check_dtype=False,rtol=1e-12,atol=1e-12)
    values=frame[["store_id","item_id","origin","horizon","target_date","model","raw_prediction","prediction"]].rename(columns={"origin":"origin_date"})
    statuses=frame[["series_id","origin","model","status","reason"]].drop_duplicates().to_dict("records")
    return values,statuses,frame,{"forecasts":fingerprint(directory/"tables/forecasts.parquet"),"summary":fingerprint(directory/"summary.json"),
        "mode":"verified source/cohort/config/cutoff/key/actual/scale-bound Season 13 reference predictions, not refitted",
        "neural_reference_fits":summary["fit_records"]}


def run(root,config):
    began=perf_counter()
    root,source,warehouse,env=settings(root); build_artifacts(root,source,warehouse)
    paths=[source,root/"data/interim/sales_long.parquet",*sorted((root/"data/raw").glob("*.csv"))]
    before={p.relative_to(root).as_posix():fingerprint(p) for p in paths}
    output=root/"data/tft/global_h28"
    with connection(warehouse,env) as con,TemporaryDirectory(prefix=".tft-",dir=root/"data") as temporary:
        temporary=Path(temporary); initial=totals(con)
        expected=json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(value)!=str(expected[name]) for name,value in initial.items()): raise ValueError("Warehouse baseline changed")
        data=prepare(con,root,config)
        reference,coverage,reference_scored,provenance=reference_forecasts(root,before,data)
        cutoff=pd.Timestamp(data["boost"].validation_origin); validation_end=cutoff+pd.Timedelta(days=config.horizon)
        if validation_end>=data["origins"][0]: raise ValueError("Validation overlaps retrospective evaluation")
        dev_rows=examples(data["train_keys"],data["training_origins"],config,cutoff)
        val_rows=examples(data["train_keys"],[cutoff],config)
        processor=Preprocessor.fit(data["panel"],dev_rows,config,cutoff)
        development=SequenceDataset(data["panel"],dev_rows,processor,config,label_cutoff=cutoff)
        validation=SequenceDataset(data["panel"],val_rows,processor,config,label_cutoff=validation_end)
        print(f"TFT real data: {len(dev_rows)} development + {len(val_rows)} validation windows; {len(data['train_keys'])} training series; {len(data['daily']):,} cached daily rows",flush=True)
        model,meta=train(development,processor,config,cutoff=cutoff,validation=validation,validation_cutoff=validation_end)
        save(model,processor,meta,temporary/"models/development")
        selected=meta["selected_epochs"]
        probe=SequenceDataset(data["panel"],val_rows,processor,config)
        reloaded,_,_=load(temporary/"models/development",device=config.device)
        np.testing.assert_allclose(predict(model,probe,config)[0],predict(reloaded,probe,config)[0],rtol=1e-6,atol=1e-6)
        records=[{**meta,"phase":"development","model_bytes":(temporary/"models/development/weights.pt").stat().st_size}]
        predictions,checks,interpretations,audits,eligibility,unknowns,curves,raw_diagnostics=[reference],[],[],[],[],[],[],{}
        print(f"Frozen December-selected TFT epoch count: {selected}",flush=True)
        for origin in [cutoff,*data["origins"]]:
            fitting_rows=examples(data["train_keys"],data["training_origins"],config,origin)
            eligibility.append({"fit_cutoff":origin,"eligible_windows":len(fitting_rows),"label_values":len(fitting_rows)*config.horizon,
                                "excluded_incomplete_vectors":len(data["train_keys"])*len(data["training_origins"])-len(fitting_rows),
                                "target_max":fitting_rows.origin_date.max()+pd.Timedelta(days=config.horizon),"reason":"target_end_after_fit_cutoff"})
            if origin==cutoff: continue
            prep=Preprocessor.fit(data["panel"],fitting_rows,config,origin)
            fitting=SequenceDataset(data["panel"],fitting_rows,prep,config,label_cutoff=origin)
            rows=examples(data["eval_keys"],[origin],config)
            inference=SequenceDataset(data["panel"],rows,prep,config)
            alternatives=[]
            for changed in future_panels(data["daily"],data["calendar"],data["metadata"],origin):
                candidate=SequenceDataset(changed,rows,prep,config); assert_same_tensors(inference,candidate)
                assert_same_tensors(fitting,SequenceDataset(changed,fitting_rows,prep,config,label_cutoff=origin))
                alternatives.append(candidate)
            model,meta=train(fitting,prep,config,cutoff=origin,epochs=selected)
            path=temporary/"models"/f"origin_{origin:%Y%m%d}"; save(model,prep,meta,path)
            tick=perf_counter(); raw,scored,diagnostics=predict(model,inference,config,diagnostics=True)
            meta["prediction_seconds"]=perf_counter()-tick
            loaded,saved_prep,_=load(path,device=config.device)
            assert saved_prep.state()==prep.state()
            maximum=0.
            for candidate in [inference,*alternatives]:
                again,pred,diagnostic=predict(loaded,candidate,config,diagnostics=True)
                np.testing.assert_allclose(raw,again,rtol=1e-6,atol=1e-6); np.testing.assert_allclose(scored,pred,rtol=1e-6,atol=1e-6)
                for name in diagnostics: np.testing.assert_allclose(diagnostics[name],diagnostic[name],rtol=1e-6,atol=1e-6)
                maximum=max(maximum,float(np.abs(raw-again).max()))
            values=prediction_rows(rows,scored,"tft"); values["raw_prediction"]=raw.reshape(-1)
            predictions.append(values)
            for member in data["evaluation_cohort"].itertuples():
                coverage.append({"model":"tft","origin":origin,"series_id":member.series_id,"status":"ok","reason":None})
            for name,values in diagnostics.items(): raw_diagnostics[f"{origin:%Y%m%d}_{name}"]=values
            raw_diagnostics[f"{origin:%Y%m%d}_store_id"]=rows.store_id.to_numpy(dtype=str)
            raw_diagnostics[f"{origin:%Y%m%d}_item_id"]=rows.item_id.to_numpy(dtype=str)
            interpretations.append(interpretation_tables(rows,diagnostics,inference.tensors[0][:,:,1].numpy()))
            audits.extend(inference.audits)
            for name,index in zip(saved_prep.vocabulary,range(len(saved_prep.vocabulary))):
                unknowns.append({"origin":origin,"field":name,"unknown_count":int((inference.tensors[2][:,index]==0).sum()),"series":len(inference)})
            checks.append({"origin":origin,"rows":raw.size,"tensors_exact":True,"future_sales_mutated_and_removed":True,
                           "future_prices_labels_excluded":True,"eligible_training_unchanged":True,"preprocessing_roundtrip_equal":True,
                           "checkpoint_roundtrip_equal":True,"vsn_probabilities_normalized":True,"attention_probabilities_normalized":True,
                           "forbidden_attention_exactly_zero":True,"max_forecast_difference":maximum,"atol":1e-6,"rtol":1e-6})
            records.append({**meta,"phase":"evaluation_refit","model_bytes":(path/"weights.pt").stat().st_size})
            print(f"Verified TFT {origin:%Y-%m-%d}: {len(fitting)} eligible windows, 364 forecasts, future invariance and attention/VSN gates passed",flush=True)
        forecasts,scales=attach_targets(pd.concat(predictions,ignore_index=True),coverage,data,MODELS)
        old=forecasts[forecasts.model.isin(BASELINES)].merge(reference_scored,on=["store_id","item_id","origin","horizon","target_date","model"],validate="one_to_one",suffixes=("_new","_old"))
        for name in ("actual","scale","prediction","raw_prediction"):
            np.testing.assert_allclose(old[f"{name}_new"].astype(float),old[f"{name}_old"].astype(float),equal_nan=True,rtol=1e-12,atol=1e-12)
        tables=comparison_tables(forecasts,MODELS)
        if len(tables["comparison_membership"])!=1092 or len(forecasts)!=8736: raise ValueError("Continuity comparison grid changed")
        for name in interpretations[0]:
            if name in ("vsn_summary","attention_lags","attention_horizon"): continue
            tables[name]=pd.concat([item[name] for item in interpretations],ignore_index=True)
        vsn=tables["vsn_cases"].query("valid_position")
        tables["vsn_summary"]=vsn.groupby(["family","variable"],as_index=False).agg(mean_weight=("weight","mean"),sd_weight=("weight","std"),positions=("weight","size"))
        attention=tables["attention_cases"]
        tables["attention_lags"]=attention[attention.key_family.eq("history")].groupby("key_relative_day",as_index=False).weight.mean()
        mass=attention.groupby(["store_id","item_id","origin","horizon","key_family"],as_index=False).weight.sum()
        tables["attention_horizon"]=mass.groupby(["horizon","key_family"],as_index=False).agg(mean_mass=("weight","mean"),sd_mass=("weight","std"),cases=("weight","size"))
        paired=paired_losses(forecasts,baseline="nbeats_interpretable")
        tables["paired_nbeats"]=paired
        cases=paired.groupby(["model","regime","series_id","origin"],as_index=False).absolute_loss_difference.mean()
        tables["segment_loss_variation"]=cases.groupby(["model","regime"],as_index=False).agg(series=("series_id","nunique"),cases=("origin","size"),mean_difference=("absolute_loss_difference","mean"),sd_case_difference=("absolute_loss_difference","std"),min_case_difference=("absolute_loss_difference","min"),max_case_difference=("absolute_loss_difference","max"))
        for record in records: curves.extend({**row,"phase":record["phase"],"fit_cutoff":record["fit_cutoff"]} for row in record["curves"])
        fits=[{k:v for k,v in record.items() if k not in ("curves","config")} for record in records]
        for record in fits:
            for key,value in list(record.items()):
                if isinstance(value,(list,dict)): record[key]=json.dumps(value)
        tables.update(forecasts=forecasts,scales=scales,fit_status=pd.DataFrame(fits),training_curves=pd.DataFrame(curves),
                      eligibility=pd.DataFrame(eligibility),tensor_audits=pd.DataFrame(audits),unknown_ids=pd.DataFrame(unknowns),
                      leakage_checks=pd.DataFrame(checks),training_cohort=data["training_cohort"],evaluation_cohort=data["evaluation_cohort"])
        if totals(con)!=initial or before!={p.relative_to(root).as_posix():fingerprint(p) for p in paths}: raise ValueError("Original sources or warehouse totals changed")
        (temporary/"tables").mkdir()
        for name,frame in tables.items(): frame.to_parquet(temporary/"tables"/f"{name}.parquet",index=False)
        np.savez_compressed(temporary/"interpretation_arrays.npz",**raw_diagnostics)
        summary={"status":"passed","config":asdict(config),"selected_epochs":selected,"training_series":len(data["train_keys"]),
                 "daily_cache_rows":len(data["daily"]),"development_windows":len(development),"validation_windows":len(validation),
                 "development_cutoff":cutoff,"validation_end":validation_end,"evaluation_origins":data["origins"],
                 "fit_records":records,"eligibility":eligibility,"leakage_checks":checks,"reference_provenance":provenance,
                 "metrics_common":tables["metrics_common"].to_dict("records"),"source_fingerprints":before,"inputs_unchanged":True,
                 "warehouse_totals":initial,"table_rows":{name:len(frame) for name,frame in tables.items()},"runtime_seconds":perf_counter()-began,
                 "versions":{name:version(name) for name in ("torch","numpy","pandas","duckdb")},
                 "resources":{"device":config.device,"threads":config.threads,"workers":0,"mixed_precision":False,
                              "largest_tensor_cache_bytes":sum(t.numel()*t.element_size() for t in fitting.tensors),"peak_process_memory":"not sampled"},
                 "limitations":["Previously examined 13-series retrospective benchmark, not untouched prospective testing or official M5 WRMSSE",
                                "Point MSE/linear head instead of original quantiles; small fixed CPU budget, no test-based hyperparameter search",
                                "Scheduled calendars assumed known; prices unavailable without as-of evidence",
                                "VSN/attention are internal diagnostics, not causal importance, stockouts or price elasticity",
                                "Covariate-enriched TFT versus target-only N-BEATS compares whole systems, not architecture alone",
                                "Small correlated segment/origin samples; reported loss variation is descriptive, not a confidence interval",
                                "Recorded sales are not verified latent demand; no inventory savings or validated cold-start accuracy"]}
        from .report import artifacts
        artifacts(summary,tables,temporary)
        for name,value in (("summary",summary),("leakage_verification",checks)):
            (temporary/f"{name}.json").write_text(json.dumps(json_ready(value),default=str,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        publish(temporary,output)
    return summary,output
