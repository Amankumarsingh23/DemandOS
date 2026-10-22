"""Read-only warehouse access; small actual-sales snapshots, never causal outcomes."""
from dataclasses import dataclass
from pathlib import Path
import json
import tomllib
import numpy as np
import pandas as pd

from demandos.data.pipeline import fingerprint
from demandos.features.pipeline import totals,load_keys
from demandos.features.config import FeatureConfig
from demandos.features.encoding import CategoryMapping
from demandos.warehouse.cli import settings,connection,build_artifacts
from .core import require,file_hash,read_json

ORIGINS = tuple(pd.Timestamp(x) for x in ["2016-01-31","2016-02-28","2016-03-27"])


@dataclass
class Inputs:
    root: Path
    warehouse: Path
    env: dict
    source_hashes: dict
    model_hashes: dict
    warehouse_totals: dict
    snapshot: pd.DataFrame
    calendar: pd.DataFrame
    metadata: pd.DataFrame
    feature_config: FeatureConfig
    mapping: CategoryMapping
    settings: dict
    reference_features: pd.DataFrame
    reference_forecasts: pd.DataFrame


def load(root):
    root,source,warehouse,env=settings(root);build_artifacts(root,source,warehouse)
    sources=[source,root/"data/interim/sales_long.parquet",*sorted((root/"data/raw").glob("*.csv"))]
    hashes={p.relative_to(root).as_posix():fingerprint(p) for p in sources}
    prior=root/"data/quantiles/global_h28";summary=read_json(prior/"summary.json")
    require(summary["status"]=="passed" and summary["source_fingerprints"]==hashes,"SOURCE_SNAPSHOT_DIGEST")
    cohort=pd.read_parquet(prior/"tables/evaluation_cohort.parquet")
    require(len(cohort)==25 and not cohort.duplicated(["store_id","item_id"]).any(),"EXPECTED_COHORT")
    model_paths=[]
    for prefix in [prior,root/"data/conformal/m5_h28"]:
        record=read_json(prefix/"summary.json")
        require(record["status"]=="passed" and record["source_fingerprints"]==hashes,"MODEL_SOURCE_PROVENANCE",prefix)
        for q in ["p10","p50","p90"]:
            for name in ["metadata.json","model.txt"]:
                p=prefix/"models/lightgbm"/q/"frozen_evaluation"/name
                require(p.is_file(),"ARTIFACT_UNAVAILABLE",p)
                model_paths.append(p)
        model_paths.append(prefix/"summary.json")
    for name in ["feature_mapping.json","tables/inference_audit.parquet","tables/forecasts.parquet","tables/evaluation_cohort.parquet"]:
        model_paths.append(prior/name)
    model_hashes={p.relative_to(root).as_posix():file_hash(p) for p in model_paths}
    raw=read_json(prior/"models/lightgbm/p50/frozen_evaluation/metadata.json")
    config=FeatureConfig(**raw["feature_config"])
    saved=read_json(prior/"feature_mapping.json");mapping=CategoryMapping(saved["fitted_through"],saved["vocabulary"])
    with connection(warehouse,env) as con:
        original=totals(con);expected=read_json(warehouse/"analytics_quality_report.json")["fact"]
        require(original["daily_rows"]==58327370 and all(str(v)==str(expected[k]) for k,v in original.items()),"WAREHOUSE_INTEGRITY")
        con.execute("create temp view feature_base_sales as select store_id,item_id,date,units_sold from main.fact_daily_sales")
        metadata=load_keys(con,ORIGINS[0],cohort)
        metadata=metadata.merge(cohort,on=["store_id","item_id"],validate="one_to_one")
        con.register("serving_keys",metadata[["store_id","item_id"]])
        snapshot=con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price
            from main.fact_daily_sales f join serving_keys k using(store_id,item_id)
            where f.date>=? and f.date<=? order by store_id,item_id,date""",[ORIGINS[0]-pd.Timedelta(days=70),ORIGINS[-1]+pd.Timedelta(days=28)]).df()
        calendar=con.execute("select * from main.dim_calendar order by date").df()
    snapshot["date"]=pd.to_datetime(snapshot.date);snapshot["available_at"]=snapshot.date.copy()
    calendar["date"]=pd.to_datetime(calendar.date)
    settings_doc=tomllib.loads((root/"configs/optimization.toml").read_text())
    operating=tomllib.loads((root/"configs/simulation.toml").read_text())["operations"]["balanced"]
    require(operating==dict(holding=.2,lost_penalty=40.,backlog_daily=2.,order_fee=5.),"FROZEN_SIMULATED_COST_CONVENTION")
    require(all(settings_doc["optimization"][k]==v for k,v in dict(pack=1,moq=0,purchase=60.,on_hand=10,on_order=3,review_days=7,seed=42).items()),"FROZEN_SIMULATED_STATE_CONVENTION")
    metadata["series_id"]=[json.dumps([s,i],separators=(",",":")) for s,i in metadata[["store_id","item_id"]].itertuples(index=False,name=None)]
    metadata["volume"]=metadata.cat_id.map(settings_doc["volume_per_unit"])
    metadata=metadata.sort_values("series_id").reset_index(drop=True)
    require(metadata.cohort.value_counts().to_dict()=={"continuity":13,"sparse_stress":12},"EXPECTED_COHORT")
    reference=pd.read_parquet(prior/"tables/inference_audit.parquet")
    forecasts=pd.read_parquet(prior/"tables/forecasts.parquet")
    forecasts=forecasts[forecasts.model.eq("lightgbm_quantiles")].copy()
    return Inputs(root,warehouse,env,hashes,model_hashes,original,snapshot,calendar,metadata,config,mapping,
                  {**settings_doc["optimization"],**operating},reference,forecasts)


def verify_sources(data):
    for relative,expected in data.source_hashes.items():
        require(fingerprint(data.root/relative)==expected,"SOURCE_CHANGED",relative)
    for relative,expected in data.model_hashes.items():
        require(file_hash(data.root/relative)==expected,"HISTORICAL_MODEL_CHANGED",relative)
    with connection(data.warehouse,data.env) as con:
        require(totals(con)==data.warehouse_totals,"WAREHOUSE_CHANGED")
