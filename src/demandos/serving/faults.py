"""Predeclared SYNTHETIC faults on real snapshots and isolated copied artifacts."""
from pathlib import Path
import shutil
import uuid
import numpy as np
import pandas as pd

from .core import Blocked,require,digest,file_hash,atomic_json
from .features import extract,validate_frame,parity,CASE
from .registry import Registry
from .prediction import validate_output
from .publication import publish,read_batch
from .decisions import plan,validate_decisions
from .monitoring import incident,drift,mature_labels,performance
from .routing import fallback
from .bootstrap import APPROVED,CANDIDATE,BASELINE
from .batch import LINEAGE,serve


def exercise(data,registry,bundle,contract_digest,config,scenarios,resources,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    scratch=directory/("injection-"+uuid.uuid4().hex[:8]);scratch.mkdir()
    origin=pd.Timestamp("2016-01-31");features=bundle["features"];f=bundle["forecasts"];d=bundle["decisions"]
    rows=[];incidents=[];shifts=[];proof=[]
    def blocked(name,fn,allowed=None):
        try:fn()
        except Blocked as exc:
            require(allowed is None or exc.signal in allowed,"WRONG_FAULT_GATE",f"{name}: {exc.signal}")
            rows.append(dict(fault=name,status="BLOCKED_AS_DESIGNED",signal=exc.signal,severity="blocking",passed=True))
            incidents.append(incident(exc.signal,"blocking",origin,model_version=APPROVED,batch_id=bundle["batch_id"],feature_digest=contract_digest,
                                      evidence=dict(fault=name,detail=exc.evidence),disposition="synthetic injection rejected; no live orders"))
            return
        raise AssertionError("Fault silently accepted: "+name)
    def source(snapshot,metadata=None):return extract(snapshot,data.calendar,data.metadata if metadata is None else metadata,origin,data.feature_config,data.mapping)
    blocked("bad_input_schema",lambda:source(data.snapshot.drop(columns="units_sold")))
    blocked("stale_source",lambda:source(data.snapshot[~data.snapshot.date.eq(origin)]))
    blocked("incomplete_history",lambda:source(data.snapshot[~data.snapshot.date.eq(origin-pd.Timedelta(days=5))]))
    blocked("duplicate_source",lambda:source(pd.concat([data.snapshot,data.snapshot.iloc[[0]]])))
    late=data.snapshot.copy();late.loc[late.date.eq(origin),"available_at"]=origin+pd.Timedelta(days=1)
    blocked("synthetic_delayed_arrival",lambda:source(late))
    meta=data.metadata.copy();meta["state_id"]="UNKNOWN"
    blocked("categorical_schema_mismatch",lambda:source(data.snapshot,meta))
    future=features.copy();future["max_sales_source_date"]+=pd.Timedelta(days=1)
    blocked("future_leaked_feature",lambda:validate_frame(future,data.feature_config,data.mapping))
    for name,column,value in [("lag_offset_mismatch","origin_lag_1",99999),("null_rule_mismatch","last_known_price",0.),("encoding_mismatch","store_id_code",9999)]:
        bad=features.copy();bad.loc[0,column]=value
        blocked(name,lambda bad=bad:parity(features,bad))
    bad=f.copy();bad.loc[0,"q50"]=np.nan
    blocked("candidate_nonfinite_output",lambda:bundle["validator"](bad,d))
    blocked("duplicate_prediction_key",lambda:bundle["validator"](pd.concat([f,f.iloc[[0]]]),d))
    bad=d.copy();bad.loc[0,"approved_quantity"]=100000000
    blocked("resource_feasibility_violation",lambda:validate_decisions(bad))
    blocked("optimizer_infeasibility",lambda:plan(f,data.metadata,data.settings,scenarios,resources,force_infeasible=True))
    blocked("wrong_source_digest",lambda:serve(registry,directory/"bad-source",APPROVED,data.snapshot,data.calendar,data.metadata,origin,
            data.feature_config,data.mapping,contract_digest,config,data.settings,scenarios,resources,expected_source_digest="incorrect"))
    sandbox=scratch/"mirror";sandbox.mkdir(exist_ok=True)
    shutil.copytree(registry.directory,sandbox/"registry",dirs_exist_ok=True)
    for entry_path in (registry.directory/"entries").glob("*.json"):
        from .core import read_json
        for relative in read_json(entry_path)["artifact_files"]:
            target=sandbox/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(data.root/relative,target)
    mirror=Registry(sandbox,sandbox/"registry",[sandbox])
    entry=mirror.entry(CANDIDATE)
    fields=["model_id","kind","training_cutoff","validation_window","feature_schema_version","feature_contract_digest","horizons","output_schema","dependencies","historical_provenance"]
    mirror.register("unapproved-probe",{k:entry[k] for k in fields},[sandbox/p for p in entry["artifact_files"]],asof=str(origin))
    blocked("unapproved_candidate_canary",lambda:mirror.verify("unapproved-probe",approved=True))
    mirror.transition("unapproved-probe","approved",reason="isolated lifecycle test",asof=str(origin))
    mirror.transition("unapproved-probe","retired",reason="isolated retirement test",asof=str(origin))
    blocked("retired_model_load",lambda:mirror.verify("unapproved-probe"))
    relative=next(p for p in mirror.entry(APPROVED)["artifact_files"] if p.endswith("model.txt"))
    artifact=sandbox/relative;original=artifact.read_bytes();artifact.write_bytes(original+b"\nTAMPERED_FAULT\n")
    blocked("modified_model_hash",lambda:mirror.verify(APPROVED))
    selected=fallback(mirror,contract_digest,origin,"ARTIFACT_DIGEST_MISMATCH")
    require(selected["mode"]=="TRUSTED_ELIGIBLE_BASELINE","FALLBACK_EXERCISE_FAILED")
    rows.append(dict(fault="model_failure_baseline_fallback",status="RECOVERED_TRUSTED_ELIGIBLE_BASELINE",signal="ARTIFACT_DIGEST_MISMATCH",severity="warning",passed=True))
    incidents.append(incident("BASELINE_FALLBACK","warning",origin,model_version=BASELINE,evidence=selected,disposition="separate validated fallback replay"))
    artifact.write_bytes(original)
    blocked("incompatible_feature_fallback",lambda:mirror.resolve("fallback",contract_digest="incompatible",origin=origin))
    require(fallback(mirror,"incompatible",origin,"ARTIFACT_DIGEST_MISMATCH")["mode"]=="BLOCKED_HUMAN_REVIEW","UNSAFE_FALLBACK")
    require(fallback(mirror,contract_digest,origin,"STALE_SOURCE")["mode"]=="BLOCKED_HUMAN_REVIEW","UNSAFE_FALLBACK")
    if mirror.state()["status"][CANDIDATE]=="candidate":mirror.transition(CANDIDATE,"approved",reason="isolated valid digest/schema approval",asof=str(origin))
    mirror.activate("forecast",CANDIDATE,contract_digest=contract_digest,reason="isolated rollback setup",asof=str(origin))
    mirror.activate("forecast",APPROVED,contract_digest=contract_digest,reason="isolated verified rollback",asof=str(origin),rollback=True)
    require(mirror.resolve("forecast")["model_version"]==APPROVED,"ROLLBACK_FAILED")
    rows.append(dict(fault="valid_rollback",status="RECOVERED_PRIOR_APPROVED_ALIAS",signal="ROLLBACK",severity="informational",passed=True))
    mirror.activate("forecast",CANDIDATE,contract_digest=contract_digest,reason="missing target setup",asof=str(origin))
    held=artifact.with_suffix(".held");artifact.rename(held)
    before=mirror.state()
    try:blocked("rollback_target_unavailable",lambda:mirror.activate("forecast",APPROVED,contract_digest=contract_digest,reason="unavailable",asof=str(origin),rollback=True))
    finally:held.rename(artifact)
    require(mirror.state()==before,"FAILED_ROLLBACK_CHANGED_ALIAS")
    mirror.activate("forecast",APPROVED,contract_digest=contract_digest,reason="restore isolated champion",asof=str(origin),rollback=True)
    artifact.write_bytes(original+b"bad")
    blocked("rollback_target_digest_mismatch",lambda:mirror.activate("forecast",APPROVED,contract_digest=contract_digest,reason="bad hash",asof=str(origin),rollback=True))
    artifact.write_bytes(original)
    corrupt=scratch/"corrupt-publication"
    shutil.copytree(Path(bundle["directory"])/"batches"/bundle["batch_id"],corrupt,dirs_exist_ok=True)
    path=corrupt/"forecasts.parquet";path.write_bytes(path.read_bytes()+b"CORRUPTED")
    blocked("corrupted_batch_publication",lambda:read_batch(corrupt))
    retry_root=scratch/"retry"
    blocked("incomplete_staging_write",lambda:publish(retry_root,bundle["request"],f,d,bundle["validator"],interrupt=True))
    batch,manifest,status=publish(retry_root,bundle["request"],f,d,bundle["validator"])
    _,same,status2=publish(retry_root,bundle["request"],f,d,bundle["validator"])
    require(status=="PUBLISHED_LOCAL_SIMULATION" and status2=="REUSED_NO_NEW_ACTIONS" and manifest==same,"RETRY_DUPLICATED_ACTIONS")
    rows.append(dict(fault="interruption_retry_and_duplicate_retry",status="RECOVERED_ONE_COMPLETED_BATCH",signal="IDEMPOTENT_RETRY",severity="informational",passed=True))
    # A valid changed forecast/plan with the SAME request is still a hard conflict.
    other=f.copy();other[["q10","q50","q90","raw_q10","raw_q50","raw_q90","prediction"]]*=1.10
    decision,_=plan(other,data.metadata,data.settings,scenarios,resources)
    for name in LINEAGE:decision[name]=d[name].iloc[0]
    blocked("idempotency_content_conflict",lambda:publish(retry_root,bundle["request"],other,decision,bundle["validator"]))
    labels=data.snapshot[["store_id","item_id","date","units_sold"]]
    removed=labels[~(labels.store_id.eq(f.store_id.iloc[0])&labels.item_id.eq(f.item_id.iloc[0])&labels.date.eq(f.target_date.iloc[0]))]
    missing=performance(mature_labels(f,removed,origin+pd.Timedelta(days=30),lag_days=2))
    require(missing["missing_labels"]==1 and missing["mature"]==699,"MISSING_LABEL_ZERO_FILLED")
    rows.append(dict(fault="missing_mature_label",status="WARN_UNAVAILABLE_NEVER_ZERO_FILLED",signal="MISSING_MATURE_LABEL",severity="warning",passed=True))
    incidents.append(incident("MISSING_MATURE_LABEL","warning",origin+pd.Timedelta(days=30),batch_id=bundle["batch_id"],evidence=missing,disposition="one metric label unavailable; forecasts preserved"))
    reference=features.drop_duplicates(["store_id","item_id"])
    for mode in ["mean_increase","zero_inflation","extreme_spike"]:
        changed=data.snapshot.copy();mask=changed.date.le(origin)&changed.date.gt(origin-pd.Timedelta(days=56))
        if mode=="mean_increase":changed.loc[mask,"units_sold"]=3*changed.loc[mask,"units_sold"]+2
        elif mode=="zero_inflation":changed.loc[changed.date.between(origin-pd.Timedelta(days=8),origin),"units_sold"]=0
        else:changed.loc[changed.date.eq(origin),"units_sold"]+=100000
        new,_=source(changed);new=new.drop_duplicates(["store_id","item_id"])
        column="rolling_zero_fraction_7_at_origin" if mode=="zero_inflation" else "rolling_mean_7_at_origin"
        result=drift(reference[column].to_numpy(float),new[column].to_numpy(float),column,config["alerts"],origin=origin)
        result["injection"]=mode;shifts.append(result)
        require(result["severity"]=="warning","SHIFT_NOT_DETECTED",mode)
        incidents.append(incident("INJECTED_DISTRIBUTION_SHIFT","warning",origin,feature_digest=contract_digest,
                                  threshold=config["alerts"],evidence=result,disposition="descriptive warning; not automatic policy failure"))
    values=reference.rolling_mean_7_at_origin.to_numpy(float)
    missing_values=values.copy();missing_values[:10]=np.nan
    result=drift(values,missing_values,"synthetic_feature_missingness",config["alerts"],origin=origin);result["injection"]="missingness";shifts.append(result)
    require(result["severity"]=="warning","MISSINGNESS_NOT_DETECTED")
    incidents.append(incident("INJECTED_FEATURE_MISSINGNESS","warning",origin,threshold=config["alerts"],evidence=result,disposition="warning; critical history gaps separately block"))
    atomic_json(directory/"faults.json",rows);atomic_json(directory/"drift_injections.json",shifts)
    return pd.DataFrame(rows),incidents,shifts,selected
