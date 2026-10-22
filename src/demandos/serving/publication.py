"""Immutable effectively-once LOCAL batches; atomic same-volume directory rename."""
import os
from pathlib import Path
import uuid
import pandas as pd

from .core import atomic_json, read_json, digest, file_hash, frame_hash, require, writer_lock, Blocked


def request_id(entry, contract_digest, source_digest, origin, cohort, config, *, channel):
    request=dict(model_artifact_digest=entry["artifact_digest"], model_version=entry["model_version"],
                 feature_contract_digest=contract_digest,source_snapshot_digest=source_digest,origin=str(origin),
                 cohort=cohort,horizons=entry["horizons"],config=config,channel=channel,implementation="local-batch-v1")
    return digest(request),request


def read_batch(directory):
    directory=Path(directory)
    require((directory/"completed.json").is_file(),"PUBLICATION_INCOMPLETE")
    manifest=read_json(directory/"completed.json")
    require(manifest["publication_digest"]==digest({k:v for k,v in manifest.items() if k!="publication_digest"}), "PUBLICATION_MANIFEST_CORRUPTION")
    require(manifest["batch_id"]==digest(manifest["request"]), "BATCH_ID_MISMATCH")
    frames={}
    for name,expected in manifest["files"].items():
        path=(directory/name).resolve()
        require(path.parent==directory.resolve() and path.is_file() and file_hash(path)==expected, "PUBLICATION_CORRUPTION",name)
        frames[path.stem]=pd.read_parquet(path)
    require(set(frames)=={"forecasts","decisions"}, "PUBLICATION_INCOMPLETE")
    for name,frame in frames.items():
        require(frame_hash(frame)==manifest["content"][name], "PUBLICATION_CONTENT_CONFLICT",name)
    return frames,manifest


def publish(directory, request, forecasts, decisions, validator, *, interrupt=False):
    directory=Path(directory);batch=digest(request);final=directory/"batches"/batch
    validator(forecasts,decisions)
    content={"forecasts":frame_hash(forecasts),"decisions":frame_hash(decisions)}
    with writer_lock(directory,"batch-"+batch+".lock"):
        if final.exists():
            _,manifest=read_batch(final)
            require(manifest["request"]==request and manifest["content"]==content, "IDEMPOTENCY_CONFLICT",batch)
            return batch,manifest,"REUSED_NO_NEW_ACTIONS"
        stage=directory/"staging"/(batch[:12]+"-"+uuid.uuid4().hex[:12]);stage.mkdir(parents=True)
        forecasts.to_parquet(stage/"forecasts.parquet",index=False)
        if interrupt:raise Blocked("INJECTED_INTERRUPTED_STAGE",str(stage))
        decisions.to_parquet(stage/"decisions.parquet",index=False)
        # Validate the persisted bytes before publishing, not only in-memory frames.
        validator(pd.read_parquet(stage/"forecasts.parquet"),pd.read_parquet(stage/"decisions.parquet"))
        manifest=dict(batch_id=batch,request=request,content=content,files={n:file_hash(stage/n) for n in ["forecasts.parquet","decisions.parquet"]},
                      local_semantics="immutable atomic completed directory; no distributed exactly-once claim")
        manifest["publication_digest"]=digest(manifest);atomic_json(stage/"completed.json",manifest)
        read_batch(stage)
        final.parent.mkdir(parents=True,exist_ok=True)
        try:os.rename(stage,final)
        except FileExistsError as exc:raise Blocked("CONCURRENT_PUBLICATION_CONFLICT",batch) from exc
    return batch,manifest,"PUBLISHED_LOCAL_SIMULATION"
