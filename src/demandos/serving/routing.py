"""Whole-pool simulated exposure and explicit failure-specific forecast fallback."""
import json
from pathlib import Path
from .core import digest,require,Blocked,canonical,writer_lock,atomic_json,read_json


def canary(pool_id,fraction,seed=42):
    require(isinstance(pool_id,str) and bool(pool_id) and 0<=fraction<=1,"CANARY_CONTRACT")
    u=int(digest(dict(domain="simulated-canary-v1",pool=pool_id,seed=seed))[:16],16)/2**64
    return dict(pool_id=pool_id,uniform=u,fraction=fraction,route="candidate" if u<fraction else "approved",
                treatment_unit="whole shared-resource pool; stable across origins",status="SIMULATED_NO_REAL_DEPLOYMENT")


def fallback(registry,contract_digest,origin,signal):
    allowed={"ARTIFACT_DIGEST_MISMATCH","ARTIFACT_UNAVAILABLE","INVALID_MODEL_OUTPUT","NONFINITE_PREDICTIONS","MODEL_NOT_APPROVED","DEPENDENCY_MISMATCH"}
    if signal not in allowed:
        return dict(mode="BLOCKED_HUMAN_REVIEW",model_version=None,reason=signal,order_publication="blocked; zero orders are not a universal safe action")
    try:
        model=registry.resolve("fallback",contract_digest=contract_digest,origin=origin)
    except Blocked as exc:
        return dict(mode="BLOCKED_HUMAN_REVIEW",model_version=None,reason=signal+"; fallback "+exc.signal,order_publication="blocked")
    return dict(mode="TRUSTED_ELIGIBLE_BASELINE",model_version=model["model_version"],reason=signal,
                order_publication="only after independent input/output/resource validation; never stale last-good reuse")


def publish_canary(directory,route,bundle,registry,contract_digest):
    """Immutable simulated decision references; forecasts remain in their original batch.

    A separate simulation namespace never applies these as extra orders to the
    approved replay's inventory state. There is no external order dispatcher.
    """
    from .decisions import validate_decisions
    model=bundle["forecasts"].model_version.iloc[0]
    registry.verify(model,contract_digest=contract_digest,origin=bundle["forecasts"].origin_date.iloc[0],approved=True)
    decisions=bundle["comparison_decisions"].copy();validate_decisions(decisions)
    request=dict(route=route,forecast_batch_id=bundle["batch_id"],model_version=model,model_digest=bundle["forecasts"].model_digest.iloc[0],
                 feature_contract_digest=contract_digest,source_snapshot_digest=bundle["source_digest"],resource_digest=decisions.resource_digest.iloc[0],
                 execution_namespace="independent_simulated_canary_no_real_orders")
    identity=digest(request)
    decisions["canary_id"]=identity;decisions["forecast_batch_id"]=bundle["batch_id"]
    decisions["source_snapshot_digest"]=bundle["source_digest"];decisions["feature_contract_digest"]=contract_digest
    decisions["fallback_mode"]="NONE";decisions["incident_ids"]="[]"
    decisions["action_id"]=[digest(dict(canary_id=identity,series_id=s)) for s in decisions.series_id]
    body=json.loads(canonical(dict(request=request,decisions=decisions.to_dict("records"))).decode())
    record=dict(**body,publication_digest=digest(body));directory=Path(directory)
    with writer_lock(directory):
        path=directory/(identity+".json")
        if path.exists():require(read_json(path)==record,"CANARY_IDEMPOTENCY_CONFLICT")
        else:atomic_json(path,record)
    return decisions,identity
