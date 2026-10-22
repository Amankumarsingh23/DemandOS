"""Register existing native checkpoints; approvals are explicitly local-demo gates."""
from importlib.metadata import version
import pandas as pd
from .registry import Registry
from .core import read_json,require

APPROVED="s16-lgb-20160131"
CANDIDATE="s17-precalibration-lgb-20151011"
BASELINE="historical-56d-v1"


def initialize(data,directory,contract_digest):
    root=data.root
    registry=Registry(root,directory,[root/"data/quantiles/global_h28",root/"data/conformal/m5_h28",root/"configs"])
    entries=[]
    for name,base,validation_start,validation_end in [
        (APPROVED,root/"data/quantiles/global_h28","2015-12-06","2016-01-03"),
        (CANDIDATE,root/"data/conformal/m5_h28","2015-09-13","2015-10-11")]:
        files=[];cutoffs=[]
        for q,alpha in [("p10",.1),("p50",.5),("p90",.9)]:
            path=base/"models/lightgbm"/q/"frozen_evaluation"
            metadata=read_json(path/"metadata.json")
            require(metadata["quantile_alpha"]==alpha and metadata["metadata"]["version"]==version("lightgbm"),"MODEL_PROVENANCE_MISMATCH")
            require(metadata["feature_config"]==read_json(root/"data/quantiles/global_h28/models/lightgbm/p50/frozen_evaluation/metadata.json")["feature_config"],"INCOMPATIBLE_FEATURE_CONTRACT")
            cutoffs.append(metadata["metadata"]["allowed_cutoff"]);files += [path/"metadata.json",path/"model.txt"]
        require(len(set(cutoffs))==1 and pd.Timestamp(validation_end)<=pd.Timestamp(cutoffs[0]),"MODEL_PROVENANCE_MISMATCH")
        entry=registry.register(name,dict(model_id="native_lightgbm_daily_quantiles",kind="lightgbm_quantiles",training_cutoff=cutoffs[0],
            validation_window=[validation_start,validation_end],feature_schema_version="existing-origin-features-v1",feature_contract_digest=contract_digest,
            horizons=list(range(1,29)),output_schema=["q10","q50","q90","prediction=nonnegative repaired q50"],dependencies={p:version(p) for p in ["lightgbm","numpy","pandas"]},
            historical_provenance=dict(source=base.relative_to(root).as_posix(),raw_checkpoints=True,conformal_correction_applied=False,
                                       approval="synthetic local demonstration; not a historical Walmart approval; no evaluation labels used")),files,asof="2016-01-31 local historical replay setup")
        entries.append(entry)
    recipe=root/"configs/serving_baseline.json"
    entries.append(registry.register(BASELINE,dict(model_id="existing_historical_empirical_baseline",kind="historical_empirical",training_cutoff=None,
        validation_window=None,feature_schema_version="existing-origin-features-v1",feature_contract_digest=contract_digest,horizons=list(range(1,29)),
        output_schema=["q10","q50","q90","prediction=nonnegative repaired q50"],dependencies={p:version(p) for p in ["numpy","pandas"]},
        historical_provenance=dict(source="existing demandos.quantiles.baseline.historical",trained_weights=False,approval="validated recipe, not a newly fitted historical model")),[recipe],asof="2016-01-31 local replay setup"))
    for name in [APPROVED,BASELINE]:
        if registry.state()["status"][name]=="candidate":
            registry.transition(name,"approved",reason="trusted native provenance / validated baseline; subsequent per-batch hard gates mandatory",asof="2016-01-31 simulated local approval")
    if registry.state()["aliases"].get("forecast")!=APPROVED:
        registry.activate("forecast",APPROVED,contract_digest=contract_digest,reason="initial local-demo champion",asof="2016-01-31")
    if registry.state()["aliases"].get("fallback")!=BASELINE:
        registry.activate("fallback",BASELINE,contract_digest=contract_digest,reason="origin-eligible history recipe only",asof="2016-01-31")
    return registry,entries
