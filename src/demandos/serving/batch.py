"""A callable batch boundary: input gates -> verified prediction -> feasible publication."""
import pandas as pd
import json
from .core import require,digest,Blocked
from .features import extract,validate_frame,CASE,KEY
from .prediction import predict,validate_output
from .decisions import plan,validate_decisions,resource_record,target_vector
from .publication import request_id,publish

LINEAGE=["batch_id","source_snapshot_digest","channel","fallback_mode","incident_ids"]


def serve(registry,directory,name,snapshot,calendar,metadata,origin,feature_config,mapping,contract_digest,run_config,settings,scenarios,resources,
          *,channel="approved",fallback_mode="NONE",incident_ids=(),interrupt=False,expected_source_digest=None):
    require(channel in {"approved","shadow","canary","fallback"},"SERVING_CHANNEL")
    approved=channel!="shadow"
    if name in {"forecast","fallback","canary"}:
        name=registry.resolve(name,contract_digest=contract_digest,origin=origin)["model_version"]
    entry=registry.verify(name,contract_digest=contract_digest,origin=origin,approved=approved)
    features,source=extract(snapshot,calendar,metadata,origin,feature_config,mapping)
    if expected_source_digest is not None:require(source==expected_source_digest,"SOURCE_SNAPSHOT_DIGEST")
    validate_frame(features,feature_config,mapping)
    cohort=metadata[KEY].to_dict("records")
    config={**run_config,"policy_version":"origin-p50-14d-target-mean-mismatch-v1","resources":resource_record(resources),"fallback_mode":fallback_mode,"incident_ids":list(incident_ids)}
    batch,request=request_id(entry,contract_digest,source,origin,cohort,config,channel=channel)
    forecast,timing=predict(registry,name,features,snapshot,contract_digest,approved=approved)
    decision,diagnostic=plan(forecast,metadata,settings,scenarios,resources)
    # Shadow computes feasible comparison plans, but publishes no active actions.
    comparison=decision.copy()
    if channel=="shadow":decision=decision.iloc[:0].copy()
    for frame in [forecast,decision]:
        frame["batch_id"]=batch;frame["source_snapshot_digest"]=source;frame["channel"]=channel
        frame["fallback_mode"]=fallback_mode;frame["incident_ids"]=json.dumps(list(incident_ids))
    def validator(f,d):
        require(set(LINEAGE).issubset(f) and set(LINEAGE).issubset(d),"PUBLICATION_LINEAGE")
        for frame in [f,d]:
            require(frame.batch_id.eq(batch).all() and frame.source_snapshot_digest.eq(source).all() and frame.channel.eq(channel).all(),"PUBLICATION_LINEAGE")
        validate_output(f.drop(columns=LINEAGE),features,entry,contract_digest)
        require((channel=="shadow" and d.empty) or (channel!="shadow" and len(d)==len(metadata)),"SHADOW_ACTIVE_ACTIONS_OR_COVERAGE")
        validate_decisions(d)
        if not d.empty:
            require(d.model_version.eq(name).all() and d.model_digest.eq(entry["artifact_digest"]).all(),"DECISION_MODEL_LINEAGE")
            targets=target_vector(f,metadata)
            require((d.target_units.to_numpy()==targets).all() and (d.request.to_numpy()==(targets-settings["on_hand"]-settings["on_order"]).clip(min=0)).all(),"DECISION_FORECAST_LINEAGE")
    published,manifest,status=publish(directory,request,forecast,decision,validator,interrupt=interrupt)
    require(published==batch,"BATCH_ID_MISMATCH")
    return dict(batch_id=batch,forecasts=forecast,decisions=decision,comparison_decisions=comparison,features=features,source_digest=source,
                publication=manifest,publication_status=status,predictor=timing,decision_diagnostic=diagnostic,validator=validator,request=request)
