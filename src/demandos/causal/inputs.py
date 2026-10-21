"""Read-only bounded M5 extraction and source-bound prior artifact validation."""
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
import numpy as np
import pandas as pd
from demandos.warehouse.cli import settings,connection,build_artifacts
from demandos.features.pipeline import totals
from demandos.data.pipeline import fingerprint
from demandos.inventory.pipeline import read_inputs,load_config as inventory_config
from demandos.inventory.experiment import plan,ORIGINS,KEY
from demandos.inventory.scenarios import targets
from demandos.optimization.pipeline import load_config as allocation_config
from demandos.risk.scenarios import planning
from demandos.risk.model import Scenarios


@dataclass
class Inputs:
    root: Path
    warehouse: Path
    env: dict
    source_hashes: dict
    provenance: dict
    warehouse_totals: dict
    metadata: pd.DataFrame
    histories: dict
    observed: dict
    scenarios: dict
    future_scenarios: dict
    targets: dict
    history_features: dict
    prefix_checks: pd.DataFrame
    settings: dict


def load(root):
    root,source,warehouse,env=settings(root);build_artifacts(root,source,warehouse)
    paths=[source,root/"data/interim/sales_long.parquet",*sorted((root/"data/raw").glob("*.csv"))]
    hashes={p.relative_to(root).as_posix():fingerprint(p) for p in paths}
    selected,bundles,labels,provenance=read_inputs(root,hashes)
    risk=root/"data/risk/m5_h28"
    previous=json.loads((risk/"summary.json").read_text())
    if previous["status"]!="passed" or previous["source_fingerprints"]!=hashes:raise ValueError("Verified source-bound Season 21 required")
    for p in [risk/"summary.json",risk/"tables/scenario_manifest.parquet"]:
        provenance["artifact_hashes"][p.relative_to(root).as_posix()]=fingerprint(p)
    manifest=pd.read_parquet(risk/"tables/scenario_manifest.parquet")
    with connection(warehouse,env) as con:
        original=totals(con);expected=json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if original["daily_rows"]!=58327370 or any(str(v)!=str(expected[k]) for k,v in original.items()):raise ValueError("Warehouse integrity changed")
        con.register("causal_keys",selected[["store_id","item_id"]])
        daily=con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price from main.fact_daily_sales f
            join causal_keys k using(store_id,item_id) where date>? and date<=? order by store_id,item_id,date""",
            [ORIGINS[0]-pd.Timedelta(days=365),ORIGINS[-1]+pd.Timedelta(days=28)]).df()
        metadata=con.execute("""select distinct f.store_id,f.item_id,f.dept_id,f.cat_id from main.fact_daily_sales f
            join causal_keys k using(store_id,item_id)""").df().merge(selected,on=["store_id","item_id"],validate="one_to_one")
    cfg20=allocation_config(root/"configs/optimization.toml")
    ops=tomllib.loads((root/"configs/simulation.toml").read_text())["operations"]["balanced"]
    if ops!=dict(holding=.2,lost_penalty=40.,backlog_daily=2.,order_fee=5.):raise ValueError("Frozen balanced operating convention changed")
    metadata["series_id"]=[json.dumps([s,i],separators=(",",":")) for s,i in metadata[["store_id","item_id"]].itertuples(index=False,name=None)]
    metadata["volume"]=metadata.cat_id.map(cfg20["volume_per_unit"])
    metadata=metadata.sort_values("series_id").reset_index(drop=True)
    if len(metadata)!=25 or metadata.cohort.value_counts().to_dict()!={"continuity":13,"sparse_stress":12}:raise ValueError("Established population required")
    histories=dict(tuple(daily.groupby(["store_id","item_id"])));label_groups=dict(tuple(labels.groupby(KEY)))
    config=inventory_config(root/"configs/inventory.toml");frozen={};alternatives={};checks=[]
    for key,bundle in bundles.items():
        paths,plans,_=plan(histories[key[:2]],key[2],bundle,config,key[:2])
        chosen=plans[plans.method.eq("historical_fractile")&plans.cost_regime.eq("balanced")&plans.lead_setting.eq("fixed_7")].iloc[0]
        frozen[key]=(paths["historical"][:8].copy(),int(chosen.target))
        for perturbation in ["mutate_future_sales","remove_future_sales","mutate_future_prices"]:
            h=histories[key[:2]].copy();future=h.date.gt(key[2])
            if perturbation=="mutate_future_sales":h.loc[future,"units_sold"]+=1000000
            elif perturbation=="remove_future_sales":h=h.loc[~future]
            else:h.loc[future,"sell_price"]=999999.
            other,decisions,_=plan(h,key[2],bundle,config,key[:2])
            pd.testing.assert_frame_equal(plans,decisions)
            for family in paths:np.testing.assert_array_equal(paths[family],other[family])
            alternatives[(key,perturbation)]=other["historical"][:8].copy()
            checks.append(dict(store_id=key[0],item_id=key[1],origin_date=key[2],check=perturbation,passed=True))
    scenarios={};future_scenarios={};vectors={};observed={};features={}
    for origin in ORIGINS:
        keys=[(m.store_id,m.item_id,origin) for m in metadata.itertuples()]
        computed=planning(np.stack([frozen[k][0] for k in keys]))
        m=manifest[manifest.origin_date.eq(origin)&manifest.planning_family.eq("baseline")]
        if len(m)!=200 or (m.availability_cutoff>origin).any():raise ValueError("Incomplete/late planning manifest")
        pivot=m.pivot(index=["store_id","item_id"],columns="scenario_id",values="protection_demand")
        d=np.stack([pivot.loc[k[:2],list(computed.ids)].to_numpy(float) for k in keys])
        p=m.drop_duplicates("scenario_id").set_index("scenario_id").loc[list(computed.ids),"probability"].to_numpy(float)
        np.testing.assert_array_equal(d,computed.demand);np.testing.assert_allclose(p,computed.probabilities,atol=1e-12)
        scenarios[origin]=Scenarios(d,p,computed.ids);vectors[origin]=np.array([frozen[k][1] for k in keys],int)
        future_scenarios[origin]={name:planning(np.stack([alternatives[(k,name)] for k in keys])) for name in
                                  ["mutate_future_sales","remove_future_sales","mutate_future_prices"]}
        observed[origin]=np.stack([targets(label_groups[k],origin).actual.to_numpy(int) for k in keys])
        features[origin]=history_summary(daily,origin)
        for name in future_scenarios[origin]:
            changed=daily.copy();future=changed.date.gt(origin)
            if name=="mutate_future_sales":changed.loc[future,"units_sold"]+=1000000
            elif name=="remove_future_sales":changed=changed.loc[~future]
            else:changed.loc[future,"sell_price"]=999999.
            if history_summary(changed,origin)!=features[origin]:raise AssertionError("Future context leakage")
            checks.append(dict(store_id="ALL",item_id="ALL",origin_date=origin,check="pool_context_"+name,passed=True))
    return Inputs(root,warehouse,env,hashes,provenance,original,metadata,histories,observed,scenarios,future_scenarios,vectors,features,
                  pd.DataFrame(checks),{**cfg20["optimization"],**ops})


def history_summary(daily,origin):
    h=daily[daily.date.le(origin)&daily.date.gt(origin-pd.Timedelta(days=365))]
    aggregate=h.groupby("date").units_sold.sum()
    return dict(history_mean=float(aggregate.mean()),history_zero_rate=float(h.units_sold.eq(0).mean()),
                history_cv=float(aggregate.std()/aggregate.mean()) if aggregate.mean() else 0.)


def verify_sources(data):
    for path,expected in data.source_hashes.items():
        if fingerprint(data.root/path)!=expected:raise AssertionError("Changed M5 source: "+path)
    for path,expected in data.provenance["artifact_hashes"].items():
        if fingerprint(data.root/path)!=expected:raise AssertionError("Changed prior model/artifact: "+path)
    with connection(data.warehouse,data.env) as con:
        if totals(con)!=data.warehouse_totals:raise AssertionError("Warehouse totals changed")
