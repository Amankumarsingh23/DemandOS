"""Frozen M5 scenario pools, coordinated simulation, diagnostics and publication."""
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from demandos.inventory.pipeline import read_inputs,load_config as load_inventory
from demandos.inventory.experiment import plan,ORIGINS,KEY
from demandos.inventory.core import Economics
from demandos.inventory.scenarios import targets
from demandos.simulation.pipeline import Sink
from demandos.simulation.metrics import run_metrics
from demandos.warehouse.cli import settings,build_artifacts,connection
from demandos.features.pipeline import totals
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.backtesting.pipeline import publish
from demandos.reconciliation.pipeline import peak_memory
from .coordinator import coordinated,origin_snapshot,METHODS
from .model import Fairness
from .examples import run_examples


def load_config(path):
    with Path(path).open("rb") as handle:c=tomllib.load(handle)
    if set(c)!={"optimization","volume_per_unit","sensitivity"}:raise ValueError("Unknown/missing optimization sections")
    s=c["optimization"]
    required={"seed","scenario_samples","solver_seconds","mip_relative_gap","feasibility_tolerance","budget_factors","main_lead","planning_source","review_days","on_hand","on_order","order_fee","purchase","supplier_factor","flow_factor","receiving_factor","pack","moq","sales_as_demand"}
    if set(s)!=required or s["sales_as_demand"] is not True or s["main_lead"]!="fixed_7" or s["planning_source"]!="historical_fractile":raise ValueError("Explicit bounded scenario configuration required")
    if s["seed"]!=42 or not 4<=s["scenario_samples"]<=64 or type(s["scenario_samples"])is not int or not 0<s["solver_seconds"]<=10 or s["mip_relative_gap"]!=0 or s["feasibility_tolerance"]!=1e-7:
        raise ValueError("Invalid solver/seed/scenario bounds")
    if s["budget_factors"]!=[.35,.7,1.4] or s["review_days"]!=7 or s["purchase"]!=60 or s["order_fee"]!=5 or s["pack"]!=1 or s["moq"]!=0 or s["on_hand"]!=10 or s["on_order"]!=3:
        raise ValueError("Predeclared main experiment assumptions changed")
    if set(c["volume_per_unit"])!={"FOODS","HOUSEHOLD","HOBBIES"} or any(not np.isfinite(v) or v<=0 for v in c["volume_per_unit"].values()):raise ValueError("Explicit positive assumed category volumes required")
    expected=["supplier_scarce","flow_scarce","fair_min","fair_disparity","case_pack","quantile_source","random_lead"]
    if c["sensitivity"]["studies"]!=expected:raise ValueError("Fixed sensitivity grid required")
    return c


def summarize(runs):
    rows=[];cols=["study","cost_scenario","budget_factor","lead_setting","method","cohort"]
    for key,g in runs.groupby(cols):
        d=g.demand_units.sum();served=g.immediate_served_units.sum()
        rows.append(dict(zip(cols,key),series_runs=len(g),cost_mean=float(g.operational_cost.mean()),cost_total=float(g.operational_cost.sum()),
            fill=float(served/d) if d else None,unweighted_fill=float(g.immediate_fill_rate.mean()) if g.immediate_fill_rate.notna().any() else None,
            daily_no_shortage=float(g.daily_no_shortage_fraction.mean()),lost_mean=float(g.lost_units.mean()),
            orders_mean=float(g.orders_placed.mean()),stock_mean=float(g.ending_stock.mean()),on_order_mean=float(g.ending_on_order.mean()),
            procurement_mean=float(g.procurement_cost.mean()),ordering_mean=float(g.ordering_cost.mean()),holding_mean=float(g.holding_cost.mean()),
            lost_cost_mean=float(g.lost_cost.mean()),status="SIMULATED_SALES_AS_DEMAND"))
    return pd.DataFrame(rows)


def run(root,config):
    start=perf_counter();root,source,warehouse,env=settings(root);build_artifacts(root,source,warehouse)
    files=[source,root/"data/interim/sales_long.parquet",*sorted((root/"data/raw").glob("*.csv"))]
    before={p.relative_to(root).as_posix():fingerprint(p) for p in files}
    selected,bundles,labels,provenance=read_inputs(root,before)
    previous=root/"data/simulation/m5_h28/summary.json"
    prior=json.loads(previous.read_text())
    if prior["status"]!="passed" or prior["source_fingerprints"]!=before:raise ValueError("Verified Season 19 source-bound artifacts required")
    provenance["artifact_hashes"][previous.relative_to(root).as_posix()]=fingerprint(previous)
    cfg18=load_inventory(root/"configs/inventory.toml")
    ops=tomllib.loads((root/"configs/simulation.toml").read_text())["operations"]
    with connection(warehouse,env) as con:
        initial=totals(con)
        expected=json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if initial["daily_rows"]!=58327370 or any(str(v)!=str(expected[k]) for k,v in initial.items()):raise ValueError("Warehouse integrity changed")
        con.register("allocation_keys",selected[["store_id","item_id"]])
        daily=con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price from main.fact_daily_sales f
            join allocation_keys k using(store_id,item_id) where date>? and date<=? order by store_id,item_id,date""",
            [ORIGINS[0]-pd.Timedelta(days=365),ORIGINS[-1]+pd.Timedelta(days=28)]).df()
        metadata=con.execute("""select distinct f.store_id,f.item_id,f.dept_id,f.cat_id from main.fact_daily_sales f
            join allocation_keys k using(store_id,item_id)""").df().merge(selected,on=["store_id","item_id"],validate="one_to_one")
    metadata["series_id"]=[json.dumps([s,i],separators=(",",":")) for s,i in metadata[["store_id","item_id"]].itertuples(index=False,name=None)]
    metadata["volume"]=metadata.cat_id.map(config["volume_per_unit"])
    metadata=metadata.sort_values("series_id").reset_index(drop=True)
    if len(metadata)!=25 or metadata.volume.isna().any():raise ValueError("Incomplete origin-safe allocation group metadata")
    history=dict(tuple(daily.groupby(["store_id","item_id"])));label_groups=dict(tuple(labels.groupby(KEY)))
    frozen={};origin_checks=[];scenario_rows=[];snapshot_alternatives={name:{} for name in ["mutate_future_sales","remove_future_sales","mutate_future_prices"]}
    for key,bundle in bundles.items():
        paths,plans,audit=plan(history[key[:2]],key[2],bundle,cfg18,key[:2]);frozen[key]=(paths,plans)
        for perturbation in ["mutate_future_sales","remove_future_sales","mutate_future_prices"]:
            altered=history[key[:2]].copy();mask=altered.date.gt(key[2])
            if perturbation=="mutate_future_sales":altered.loc[mask,"units_sold"]+=1000000
            elif perturbation=="remove_future_sales":altered=altered.loc[~mask]
            else:altered.loc[mask,"sell_price"]=999999.
            other,decisions,_=plan(altered,key[2],bundle,cfg18,key[:2])
            pd.testing.assert_frame_equal(plans,decisions)
            for name in paths:np.testing.assert_array_equal(paths[name],other[name])
            row=decisions[decisions.method.eq("historical_fractile")&decisions.cost_regime.eq("balanced")&decisions.lead_setting.eq("fixed_7")].iloc[0]
            snapshot_alternatives[perturbation][key]=(int(row.target),other["historical"][:config["optimization"]["scenario_samples"],:14].sum(axis=1))
            origin_checks.append(dict(zip(KEY,key),test=perturbation,passed=True))
    cases=[];s=config["optimization"];sens=config["sensitivity"]
    snapshots=[]
    snapshot_baseline={}
    for key,(paths,plans) in frozen.items():
        chosen=plans[plans.method.eq("historical_fractile")&plans.cost_regime.eq("balanced")&plans.lead_setting.eq("fixed_7")].iloc[0]
        snapshot_baseline[key]=(int(chosen.target),paths["historical"][:s["scenario_samples"],:14].sum(axis=1))
    for origin in ORIGINS:
        baseline=None
        for perturbation,values in {"baseline":snapshot_baseline,**snapshot_alternatives}.items():
            eligible=[values[(m.store_id,m.item_id,origin)] for m in metadata.itertuples()]
            result=origin_snapshot(metadata,[x[0] for x in eligible],np.stack([x[1] for x in eligible]),s,Economics(100,60,20))
            if baseline is None:baseline=result["quantity"]
            else:np.testing.assert_array_equal(baseline,result["quantity"])
            for i,(series,quantity) in enumerate(zip(metadata.series_id,result["quantity"])):snapshots.append(dict(origin_date=origin,simulation_day=0,series_id=series,
                perturbation=perturbation,approved=int(quantity),frozen_target=eligible[i][0],solver_status=result["status"],purpose="NON_COMMITTED_ORIGIN_SNAPSHOT",passed=True))
    for cost in cfg18["costs"]:
        for factor in s["budget_factors"]:cases.append(dict(study="main",cost=cost,budget=factor,lead=cfg18["leads"][1],supplier=s["supplier_factor"],flow=s["flow_factor"],pack=1,moq=0,source="historical_fractile",fairness=Fairness()))
    for study in sens["studies"]:
        case=dict(study=study,cost=cfg18["costs"][0],budget=sens["budget_factor"],lead=cfg18["leads"][1],supplier=s["supplier_factor"],flow=s["flow_factor"],pack=1,moq=0,source="historical_fractile",fairness=Fairness())
        if study=="supplier_scarce":case["supplier"]=sens["supplier_scarce_factor"]
        if study=="flow_scarce":case["flow"]=sens["flow_scarce_factor"]
        if study=="fair_min":case["fairness"]=Fairness(minimum_share=sens["minimum_share"],soft=True,penalty=sens["fairness_soft_penalty"])
        if study=="fair_disparity":case["fairness"]=Fairness(maximum_disparity=sens["maximum_disparity"],soft=True,penalty=sens["fairness_soft_penalty"])
        if study=="case_pack":case.update(pack=sens["pack"],moq=sens["moq"])
        if study=="quantile_source":case["source"]="s16_quantile_paths"
        if study=="random_lead":case["lead"]=cfg18["leads"][2]
        cases.append(case)
    target=root/"data/optimization/m5_h28";run_rows=[];solver_frames=[];matrices=[];temporal=[];cache={}
    with TemporaryDirectory(prefix=".optimization-",dir=root/"data") as tmp:
        tmp=Path(tmp);sink=Sink(tmp/"tables")
        try:
            for origin in ORIGINS:
                actual=np.stack([targets(label_groups[(m.store_id,m.item_id,origin)],origin).actual.to_numpy(int) for m in metadata.itertuples()])
                for case_index,case in enumerate(cases):
                    cost=case["cost"];econ=Economics(**{k:v for k,v in cost.items() if k!="name"});lead=case["lead"]
                    period=max(lead["days"])+s["review_days"]
                    targets_vector=[];scenario=[]
                    for m in metadata.itertuples():
                        paths,plans=frozen[(m.store_id,m.item_id,origin)]
                        chosen=plans[plans.method.eq(case["source"])&plans.cost_regime.eq(cost["name"])&plans.lead_setting.eq(lead["name"])]
                        targets_vector.append(int(chosen.target.iloc[0]))
                        family="historical" if case["source"]=="historical_fractile" else case["source"]
                        # Random L independent of demand; sample its finite support by
                        # deterministic inverse-CDF strata, shared across all series.
                        lengths=np.array(lead["days"])[np.searchsorted(np.cumsum(lead["probabilities"]),(np.arange(s["scenario_samples"])+.5)/s["scenario_samples"])]+s["review_days"]
                        values=np.array([paths[family][w,:length].sum() for w,length in enumerate(lengths)])
                        scenario.append(values)
                        for w,value in enumerate(values):scenario_rows.append(dict(store_id=m.store_id,item_id=m.item_id,origin_date=origin,source=case["source"],lead_setting=lead["name"],scenario_id=w,probability=1/s["scenario_samples"],protection_demand=int(value),availability_cutoff=origin,scenario_dependence="independent_series_bootstraps_shared_hypothetical_lead",status="SIMULATED_SCENARIO_ASSUMED"))
                    scenario=np.stack(scenario);settings_case={**s,**ops[cost["name"]],
                        "resource_targets":[snapshot_baseline[(m.store_id,m.item_id,origin)][0] for m in metadata.itertuples()]}
                    caseid=f"{case['study']}:{cost['name']}:{case['budget']}:{lead['name']}"
                    for method in METHODS:
                        result=coordinated(actual,origin,metadata,targets_vector,scenario,settings_case,method=method,costs=econ,lead=lead,
                            budget_factor=case["budget"],supplier_factor=case["supplier"],flow_factor=case["flow"],pack=case["pack"],moq=case["moq"],fairness=case["fairness"])
                        meta=dict(origin_date=origin,scenario_config_id=caseid,allocation_group="SELECTED_COHORT_DC",study=case["study"],method=method,
                            policy_id=method,cost_scenario=cost["name"],budget_factor=case["budget"],lead_setting=lead["name"],forecast_source=case["source"],
                            forecast_availability_cutoff=origin,model_fit_cutoff=pd.Timestamp("2016-01-31") if case["source"]=="s16_quantile_paths" else origin,
                            fairness_minimum=case["fairness"].minimum_share,fairness_max_disparity=case["fairness"].maximum_disparity,
                            fairness_soft=case["fairness"].soft,scenario_status="SIMULATED_SCENARIO_ASSUMED")
                        for name in ["daily","events","allocations","resources"]:sink.write(name,result[name].assign(**meta))
                        solver_frames.append(result["solver"].assign(**meta));matrices.extend({**meta,**row} for row in result["matrices"])
                        for series,g in result["daily"].groupby("series_id",sort=False):
                            row=metadata[metadata.series_id.eq(series)].iloc[0]
                            run_rows.append({**meta,"series_id":series,"store_id":row.store_id,"item_id":row.item_id,"cohort":row.cohort,**run_metrics(g),"status":"SIMULATED_SCENARIO_ASSUMED"})
                        if case["study"]=="main" and cost["name"]=="balanced" and case["budget"]==.7 and method=="stochastic_milp":
                            cache[origin]=(actual.copy(),targets_vector,scenario.copy(),settings_case,econ,lead,result)
                    if case_index%4==0:print(f"Coordinated {origin.date()} case {case_index+1}/{len(cases)}; resource/engine invariants pass",flush=True)
            for origin,(actual,targs,scenario,settings_case,econ,lead,base) in cache.items():
                for k in [10,20]:
                    altered=actual.copy();altered[0,k-1]+=10000
                    other=coordinated(altered,origin,metadata,targs,scenario,settings_case,method="stochastic_milp",costs=econ,lead=lead)
                    for name,cols in [("allocations",["series_id","day_index","candidate_quantity","approved","position"]),("resources",["day_index","budget_before","committed","budget_after"])]:
                        a=base[name];b=other[name]
                        pd.testing.assert_frame_equal(a.loc[a.day_index.lt(k),cols].reset_index(drop=True),b.loc[b.day_index.lt(k),cols].reset_index(drop=True))
                    temporal.append(dict(origin_date=origin,mutation_day=k,test="all_series_pre_k_allocations_states_budget",passed=True))
                first=coordinated(np.full_like(actual,10000),origin,metadata,targs,scenario,settings_case,method="stochastic_milp",costs=econ,lead=lead)
                second=coordinated(np.full_like(actual,100000),origin,metadata,targs,scenario,settings_case,method="stochastic_milp",costs=econ,lead=lead)
                pd.testing.assert_frame_equal(first["allocations"][["approved","position"]],second["allocations"][["approved","position"]])
                temporal.append(dict(origin_date=origin,mutation_day=1,test="hidden_lost_demand_not_exposed",passed=True))
        finally:sink.close()
        runs=pd.DataFrame(run_rows);metrics=summarize(runs);solver=pd.concat(solver_frames,ignore_index=True)
        pair_keys=["origin_date","scenario_config_id","series_id"]
        reference=runs[runs.method.eq("proportional")][pair_keys+["operational_cost","immediate_fill_rate"]]
        paired=runs.merge(reference,on=pair_keys,validate="many_to_one",suffixes=("","_proportional"))
        if len(paired)!=len(runs) or runs.duplicated(pair_keys+["method"]).any():raise AssertionError("Unmatched optimized policy cases")
        paired["cost_difference"]=paired.operational_cost-paired.operational_cost_proportional
        generated=dict(runs=runs,metrics=metrics,solver_diagnostics=solver,paired_runs=paired,
            origin_checks=pd.DataFrame(origin_checks),temporal_checks=pd.DataFrame(temporal),
            origin_allocations=pd.DataFrame(snapshots),
            scenarios=pd.DataFrame(scenario_rows).drop_duplicates(["store_id","item_id","origin_date","source","lead_setting","scenario_id"]),metadata=metadata)
        pair_groups=["origin_date","study","cost_scenario","budget_factor","lead_setting","method","cohort"]
        generated["paired_metrics"]=paired.groupby(pair_groups,as_index=False).agg(series_origins=("series_id","size"),mean_cost_difference=("cost_difference","mean"))
        for frame in generated.values():
            if "scenario_status" not in frame:frame["scenario_status"]="SIMULATED_SCENARIO_ASSUMED"
        for name,frame in generated.items():frame.to_parquet(tmp/"tables"/(name+".parquet"),index=False)
        examples=run_examples();matrix=examples.pop("matrix");(tmp/"matrices").mkdir()
        save_npz(tmp/"matrices/toy_two_stage.npz",matrix["A"])
        (tmp/"matrix_manifest.json").write_text(json.dumps(json_ready(matrices),default=str,indent=2),encoding="utf-8")
        (tmp/"examples.json").write_text(json.dumps(json_ready(examples),default=str,indent=2),encoding="utf-8")
        from .verification import verify_outputs
        verification=verify_outputs(tmp)
        if before!={p.relative_to(root).as_posix():fingerprint(p) for p in files}:raise ValueError("Source files changed")
        if any(fingerprint(root/p)!=fp for p,fp in provenance["artifact_hashes"].items()):raise ValueError("Prior model/artifact modified")
        with connection(warehouse,env) as con:
            if totals(con)!=initial:raise ValueError("Warehouse totals changed")
        summary=dict(status="passed",scenario_status="SIMULATED_SCENARIO_ASSUMED",config=config,source_fingerprints=before,provenance=provenance,
            warehouse_totals=initial,inputs_unchanged=True,unique_real_paths=75,unique_real_daily_observations=2100,
            cases_per_origin=len(cases),pool_runs=len(cases)*3*len(METHODS),series_runs=len(runs),
            daily_rows=sink.rows["daily"],table_rows={**sink.rows,**{k:len(v) for k,v in generated.items()}},
            verification=verification,solver_status=solver.status.value_counts().to_dict(),
            solver_seconds=float(solver.runtime.sum()),maximum_mip_gap=float(solver.gap.max()) if solver.gap.notna().any() else None,
            fallbacks=int(solver.fallback.notna().sum()),examples=examples,origin_checks=len(origin_checks),temporal_checks=len(temporal),
            versions={p:version(p) for p in ["scipy","numpy","pandas","pyarrow","duckdb"]},
            warnings=["All budgets, supplier limits, volumes, acquisition costs and outcomes are scenario assumptions; no real Walmart savings.",
                "Unconstrained reference may violate shared resources and is excluded from constrained-feasibility claims.",
                "Storage capacity means per-epoch procurement-flow volume, not verified physical on-hand space.",
                "Forecast scenarios use independent per-series bootstraps with a shared assumed lead-time shock; no known joint distribution is claimed.",
                "First-stage order vector is shared; shortage/excess is scenario accounting, not adaptive multistage purchasing.",
                "Optimization mismatch/benefit and realized procurement-inclusive operating cost are different objectives.",
                "Fairness preferences are soft in the live sensitivity grid; hard resource limits are never relaxed."],skips=[])
        from .report import report
        report(summary,metrics,tmp)
        summary.update(runtime_seconds=perf_counter()-start,peak_process_mib=peak_memory())
        (tmp/"summary.json").write_text(json.dumps(json_ready(summary),default=str,indent=2,allow_nan=False),encoding="utf-8")
        (tmp/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
        publish(tmp,target)
    return summary,target
