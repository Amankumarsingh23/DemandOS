"""Digest verification precedes all native model loading; no pickle/joblib loader."""
from importlib.metadata import version
from time import perf_counter
import numpy as np
import pandas as pd

from demandos.quantiles.tree import QuantileLightGBM
from demandos.quantiles.baseline import historical
from demandos.quantiles.contracts import repair, Q
from .features import CASE, KEY, eligible_history
from .core import require, digest, read_json, STATUS


def predict(registry, name, features, snapshot, contract_digest, *, approved=False):
    tick = perf_counter(); origin = pd.Timestamp(features.origin_date.iloc[0])
    entry = registry.verify(name, contract_digest=contract_digest, origin=origin, approved=approved)
    result = features[CASE].reset_index(drop=True).copy()
    if entry["kind"] == "lightgbm_quantiles":
        columns = []
        for quantile in ["p10", "p50", "p90"]:
            paths = [registry.root/p for p in entry["artifact_files"] if p.endswith(f"/{quantile}/frozen_evaluation/metadata.json")]
            require(len(paths)==1, "MODEL_ENSEMBLE_COMPONENT", quantile)
            raw = read_json(paths[0])
            require(raw["metadata"]["version"]==version("lightgbm"), "DEPENDENCY_MISMATCH", "historical native LightGBM")
            require(raw["metadata"]["allowed_cutoff"]==entry["training_cutoff"], "MODEL_PROVENANCE_MISMATCH")
            require(raw["model_file"]=="model.txt" and str(paths[0].parent.relative_to(registry.root)/"model.txt").replace("\\","/") in entry["artifact_files"], "UNVERIFIED_MODEL_COMPONENT")
            model = QuantileLightGBM.load(paths[0].parent)
            columns.append(model.predict(features).raw_prediction.to_numpy(float))
        raw_values = np.column_stack(columns)
    else:
        recipe_paths = [registry.root/p for p in entry["artifact_files"] if p.endswith("serving_baseline.json")]
        require(len(recipe_paths)==1, "BASELINE_RECIPE")
        recipe = read_json(recipe_paths[0]); require(recipe["days"]==56 and recipe["trained_weights"] is False and recipe["algorithm"]=="historical_empirical"
            and recipe["levels"]==[.1,.5,.9] and recipe["method"]=="inverted_cdf_including_zeros", "BASELINE_RECIPE")
        snapshot=eligible_history(snapshot,origin)
        values=[]
        for store,item,horizon in features[[*KEY,"horizon"]].itertuples(index=False,name=None):
            h=snapshot[snapshot.store_id.eq(store)&snapshot.item_id.eq(item)]
            values.append(historical(h,origin,recipe["days"])[0][horizon-1])
        raw_values=np.array(values)
    require(raw_values.shape==(len(result),3) and np.isfinite(raw_values).all(), "INVALID_MODEL_OUTPUT")
    result[["raw_"+q for q in Q]] = raw_values
    result[Q] = repair(raw_values)
    result["prediction"] = result.q50
    result["model_id"],result["model_version"],result["model_digest"] = entry["model_id"],name,entry["artifact_digest"]
    result["feature_contract_digest"] = contract_digest
    result["prediction_id"] = [digest(dict(model=name,origin=origin,store=s,item=i,horizon=int(h))) for s,i,h in result[[*KEY,"horizon"]].itertuples(index=False,name=None)]
    result["scenario_status"] = STATUS
    validate_output(result,features,entry,contract_digest)
    return result,dict(seconds=perf_counter()-tick,rows=len(result),raw_crossings=int((np.diff(raw_values,axis=1)<0).any(axis=1).sum()),
                       repair="existing Season16 nonnegative monotone repair",reproducibility="absolute tolerance 1e-10; same local CPU environment")


def validate_output(frame, features, entry, contract_digest):
    required = {*CASE,*Q,"raw_q10","raw_q50","raw_q90","prediction","model_id","model_version","model_digest","feature_contract_digest","prediction_id","scenario_status"}
    require(set(frame)==required, "OUTPUT_SCHEMA")
    require(len(frame)==len(features) and not frame.duplicated(CASE).any() and frame[CASE].notna().all().all(), "DUPLICATE_OR_INCOMPLETE_PREDICTIONS")
    require(set(map(tuple,frame[CASE].to_numpy()))==set(map(tuple,features[CASE].to_numpy())), "PREDICTION_KEY_ALIGNMENT")
    require(frame.target_date.eq(frame.origin_date+pd.to_timedelta(frame.horizon,unit="D")).all(), "PREDICTION_TARGET_DATE")
    require(frame.model_version.eq(entry["model_version"]).all() and frame.model_digest.eq(entry["artifact_digest"]).all(), "PREDICTION_MODEL_LINEAGE")
    require(frame.feature_contract_digest.eq(contract_digest).all(), "PREDICTION_FEATURE_LINEAGE")
    values = frame[[*Q,"prediction","raw_q10","raw_q50","raw_q90"]]
    require(all(pd.api.types.is_numeric_dtype(x) for _,x in values.items()) and np.isfinite(values.to_numpy(float)).all(), "NONFINITE_PREDICTIONS")
    require(frame[Q].ge(0).all().all() and frame.q10.le(frame.q50).all() and frame.q50.le(frame.q90).all() and frame.prediction.eq(frame.q50).all(), "INVALID_QUANTILE_ORDER")
    require(np.array_equal(frame[Q].to_numpy(float),repair(frame[["raw_"+q for q in Q]].to_numpy(float))),"PREDICTION_REPAIR_MISMATCH")
    expected=[digest(dict(model=entry["model_version"],origin=pd.Timestamp(o),store=s,item=i,horizon=int(h))) for s,i,o,h in frame[[*KEY,"origin_date","horizon"]].itertuples(index=False,name=None)]
    require(frame.prediction_id.tolist()==expected and frame.scenario_status.eq(STATUS).all(),"PREDICTION_ID_OR_STATUS")
    require(not frame.prediction_id.duplicated().any(), "DUPLICATE_PREDICTION_ID")
