"""Bounded real M5-informed risk experiment, with simulated economic outcomes."""
from importlib.metadata import version
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from demandos.inventory.pipeline import read_inputs, load_config as inventory_config
from demandos.inventory.experiment import plan, ORIGINS, KEY
from demandos.inventory.scenarios import bootstrap, seed_for, targets
from demandos.inventory.core import Economics
from demandos.optimization.pipeline import load_config as allocation_config
from demandos.optimization.coordinator import coordinated
from demandos.optimization.model import Candidates, Resources, Fairness, fair_slacks
from demandos.simulation.pipeline import Sink
from demandos.simulation.engine import Supplier
from demandos.simulation.metrics import run_metrics
from demandos.warehouse.cli import settings, connection, build_artifacts
from demandos.features.pipeline import totals
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.backtesting.pipeline import publish
from demandos.reconciliation.pipeline import peak_memory
from .model import solve, losses
from .metrics import tail, epigraph_cvar, frontier, worst_probability
from .scenarios import planning, shifted, Closure, FAMILIES, uncertainty_box

STATUS = "SIMULATED_SCENARIO_ASSUMED"
METHODS = ["mean","cvar_0.5","cvar_2","minimax","regret","dro"]


def load_config(path):
    c = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    r = c.get("risk", {})
    expected = dict(seed=42,planning_scenarios=8,replications=8,solver_seconds=3.,alpha=.75,
                    confidence_levels=[.75,.90],penalties=[.5,2.],shock_probability=.1,
                    probability_interval=[.05,.20],risk_limits=[0.,10000.,100000.,1000000.],
                    families=list(FAMILIES),replicated_families=["baseline","joint"],methods=METHODS)
    resources = dict(budget_factor=.7,supplier_factor=.75,flow_factor=.8,closure_start=8,closure_end=14)
    if set(c) != {"risk","resources"} or r != expected or c["resources"] != resources:
        raise ValueError("Explicit predeclared bounded Season 21 experiment configuration required")
    return c


def method_options(method, alpha=.75):
    if method.startswith("cvar_"):
        return dict(method="mean_cvar",alpha=alpha,penalty=float(method.split("_")[1]))
    return dict(method=method,alpha=alpha)


def origin_problem(metadata, target, settings, budget=.7, supplier=.75, flow=.8):
    n = len(metadata);stock=np.full(n,settings["on_hand"]+settings["on_order"],float)
    req=np.maximum(0,np.asarray(target)-stock).astype(int)
    c=Candidates(tuple(metadata.series_id),req,np.full(n,settings["purchase"]),metadata.volume.to_numpy(float),
                 np.ones(n),tuple(metadata.dept_id),tuple(metadata.store_id),np.full(n,settings["order_fee"]),
                 np.ones(n),np.zeros(n),np.zeros(n))
    r=Resources(budget*(4*settings["purchase"]*sum(target)+4*n*settings["order_fee"]),max(1.,float(c.volume@target)*flow),
                {s:float(4*np.asarray(target)[np.array(c.suppliers)==s].sum()*supplier) for s in set(c.suppliers)},
                {s:max(1.,float(np.asarray(target)[np.array(c.stores)==s].sum())) for s in set(c.stores)})
    return c,r,stock


def summarize_operating(runs):
    """Pool/cohort-origin distributions across distinct complete worlds, not days."""
    group=["origin_date","study","scenario_family","cost_regime","budget_factor","method_id","world_kind","replication_id","cohort"]
    expanded=pd.concat([runs,runs.assign(cohort="ALL")],ignore_index=True)
    pools=expanded.groupby(group,dropna=False).agg(operational_cost=("operational_cost","sum"),
        demand_units=("demand_units","sum"),served_units=("immediate_served_units","sum"),series_runs=("series_id","size")).reset_index()
    pools["fill"]=pools.served_units/pools.demand_units.replace(0,np.nan)
    rows=[]
    for key,g in pools.groupby([c for c in group if c != "replication_id"],dropna=False):
        meta=dict(zip([c for c in group if c != "replication_id"],key))
        if meta["world_kind"] == "prefix_bootstrap":
            if len(g)!=8:raise AssertionError("Incomplete independent scenario-replication panel")
            p=np.full(len(g),1/len(g))
            risk=tail(g.operational_cost,p,.75)
            np.testing.assert_allclose(risk["cvar"],epigraph_cvar(g.operational_cost,p,.75)["cvar"],atol=1e-6)
            rows.append({**meta,**risk,"replications":len(g),"fill":float(g.served_units.sum()/g.demand_units.sum()) if g.demand_units.sum() else None,
                         "risk_basis":"eight distinct whole-pool hypothetical paths conditional on origin; NOT independent days or series"})
    ref=pools[pools.method_id.eq("mean")].drop(columns="method_id")
    match=[c for c in group if c != "method_id"]
    paired=pools.merge(ref,on=match,suffixes=("","_mean"),validate="many_to_one")
    if len(paired)!=len(pools):raise AssertionError("Unmatched method evaluation keys")
    paired["cost_difference_vs_mean"]=paired.operational_cost-paired.operational_cost_mean
    paired["fill_difference_vs_mean"]=paired.fill-paired.fill_mean
    return pools,pd.DataFrame(rows),paired


def run(root, config):
    started=perf_counter();root,source,warehouse,env=settings(root)
    executions=[]
    def optimize(*args,**kwargs):
        result=solve(*args,**kwargs)
        executions.append(dict(kind="risk",method=kwargs.get("method","mean"),status=result["status"],gap=result.get("mip_gap"),runtime=result["runtime"]))
        for attempt in result.get("solver_attempts",[])[:-1]:
            executions.append(dict(kind="recovered_attempt",method=kwargs.get("method","mean"),status=attempt["status"],gap=None,runtime=attempt["runtime"]))
        for oracle in result.get("oracle_solves",[]):
            executions.append(dict(kind="oracle",method="scenario_hindsight",status=oracle["status"],gap=oracle.get("mip_gap"),runtime=oracle["runtime"]))
        return result
    build_artifacts(root,source,warehouse)
    files=[source,root/"data/interim/sales_long.parquet",*sorted((root/"data/raw").glob("*.csv"))]
    before={p.relative_to(root).as_posix():fingerprint(p) for p in files}
    selected,bundles,labels,provenance=read_inputs(root,before)
    for season,path in [(19,"simulation"),(20,"optimization")]:
        p=root/f"data/{path}/m5_h28/summary.json";prior=json.loads(p.read_text())
        if prior["status"] != "passed" or prior["source_fingerprints"] != before:
            raise ValueError(f"Verified source-bound Season {season} output required")
        provenance["artifact_hashes"][p.relative_to(root).as_posix()]=fingerprint(p)
    cfg18=inventory_config(root/"configs/inventory.toml")
    cfg20=allocation_config(root/"configs/optimization.toml")
    ops=tomllib.loads((root/"configs/simulation.toml").read_text())["operations"]
    with connection(warehouse,env) as con:
        initial=totals(con)
        expected=json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if initial["daily_rows"]!=58327370 or any(str(v)!=str(expected[k]) for k,v in initial.items()):
            raise ValueError("Warehouse source integrity changed")
        con.register("risk_keys",selected[["store_id","item_id"]])
        daily=con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price from main.fact_daily_sales f
            join risk_keys k using(store_id,item_id) where date>? and date<=? order by store_id,item_id,date""",
            [ORIGINS[0]-pd.Timedelta(days=365),ORIGINS[-1]+pd.Timedelta(days=28)]).df()
        metadata=con.execute("""select distinct f.store_id,f.item_id,f.dept_id,f.cat_id from main.fact_daily_sales f
            join risk_keys k using(store_id,item_id)""").df().merge(selected,on=["store_id","item_id"],validate="one_to_one")
    metadata["series_id"]=[json.dumps([s,i],separators=(",",":")) for s,i in metadata[["store_id","item_id"]].itertuples(index=False,name=None)]
    metadata["volume"]=metadata.cat_id.map(cfg20["volume_per_unit"])
    metadata=metadata.sort_values("series_id").reset_index(drop=True)
    if len(metadata)!=25 or metadata.volume.isna().any():raise ValueError("Established 25-series population required")
    histories=dict(tuple(daily.groupby(["store_id","item_id"])));label_groups=dict(tuple(labels.groupby(KEY)))
    frozen={};mutations={};origin_checks=[]
    for key,bundle in bundles.items():
        paths,plans,_=plan(histories[key[:2]],key[2],bundle,cfg18,key[:2]);frozen[key]=(paths,plans)
        mutations[key]={}
        for name in ["mutate_future_sales","remove_future_sales","mutate_future_prices"]:
            h=histories[key[:2]].copy();future=h.date.gt(key[2])
            if name=="mutate_future_sales":h.loc[future,"units_sold"]+=1000000
            elif name=="remove_future_sales":h=h.loc[~future]
            else:h.loc[future,"sell_price"]=999999.
            other,decisions,_=plan(h,key[2],bundle,cfg18,key[:2])
            pd.testing.assert_frame_equal(plans,decisions)
            for family in paths:np.testing.assert_array_equal(paths[family],other[family])
            mutations[key][name]=(other,decisions)
            origin_checks.append(dict(zip(KEY,key),check=name,passed=True))
    rconfig=config["risk"];resources=config["resources"]
    target=root/"data/risk/m5_h28"
    run_rows=[];solver_rows=[];loss_rows=[];snapshot_rows=[];frontier_rows=[];manifests=[];boxes=[]
    temporal=[];matrices=[];shock_rows=[];static_rows=[];proof_rows=[];eval_hashes=[]
    with TemporaryDirectory(prefix=".risk-",dir=root/"data") as temporary:
        tmp=Path(temporary);sink=Sink(tmp/"tables")
        try:
            for origin in ORIGINS:
                series_keys=[(m.store_id,m.item_id,origin) for m in metadata.itertuples()]
                path_array=np.stack([frozen[k][0]["historical"] for k in series_keys])
                def targets_for(cost, source=frozen):
                    vector=[]
                    for k in series_keys:
                        plans=source[k][1]
                        row=plans[plans.method.eq("historical_fractile")&plans.cost_regime.eq(cost)&plans.lead_setting.eq("fixed_7")].iloc[0]
                        vector.append(int(row.target))
                    return np.asarray(vector,int)
                balanced_target=targets_for("balanced")
                settings_base={**cfg20["optimization"],**ops["balanced"],"resource_targets":balanced_target}
                planning_sets={name:planning(path_array,name,rconfig["shock_probability"]) for name in ["baseline","shifted_probability"]}
                for name,scenario in planning_sets.items():
                    boxes.append(dict(origin_date=origin,scenario_family=name,**uncertainty_box(scenario)))
                    for i,m in enumerate(metadata.itertuples()):
                        for j,scenario_id in enumerate(scenario.ids):
                            manifests.append(dict(store_id=m.store_id,item_id=m.item_id,origin_date=origin,availability_cutoff=origin,
                                planning_family=name,scenario_id=scenario_id,probability=scenario.probabilities[j],protection_demand=scenario.demand[i,j],
                                assumed_lead_days=7 if j<4 else 14,review_days=7,dependence="independent series blocks; common adverse-family indicator",status=STATUS))
                # Exact static origin objectives and epsilon/frontier diagnostics.
                c,r,stock=origin_problem(metadata,balanced_target,settings_base)
                scenario=planning_sets["baseline"]
                for alpha in rconfig["confidence_levels"]:
                    solutions={}
                    for method in METHODS:
                        result=optimize(c,r,scenario,stock,**method_options(method,alpha),time_limit=rconfig["solver_seconds"])
                        if result["quantity"] is None:raise AssertionError(f"No static risk solution {origin} {method} {alpha}: {result}")
                        solutions[method]=result
                    oracle=np.asarray(solutions["regret"]["oracle"])
                    for method,result in solutions.items():
                        scores=tail(result["losses"],scenario.probabilities,alpha)
                        np.testing.assert_allclose(scores["cvar"],epigraph_cvar(result["losses"],scenario.probabilities,alpha)["cvar"],atol=1e-5)
                        served=np.minimum(scenario.demand,stock[:,None]+result["quantity"][:,None]).sum(axis=0)
                        total_demand=scenario.demand.sum(axis=0)@scenario.probabilities
                        fair=fair_slacks(c,result["quantity"],Fairness())
                        normal_mean=float(np.mean(result["losses"][:4]));shock_mean=float(np.mean(result["losses"][4:]))
                        ambiguous=worst_probability(normal_mean,shock_mean)
                        frontier_rows.append(dict(origin_date=origin,method_id=method,**scores,max_regret=float((np.asarray(result["losses"])-oracle).max()),
                            objective=result["objective"],status=result["status"],gap=result["mip_gap"],ordered_units=int(sum(result["quantity"])),
                            quantities=json.dumps(result["quantity"].tolist()),fairness="none",risk_limit=None,
                            allocation_disparity=fair["disparity"],store_shares=json.dumps(fair["shares"]),
                            normal_conditional_mean=normal_mean,shock_conditional_mean=shock_mean,
                            worst_endpoint_expectation=ambiguous["cost"],worst_shock_probability=ambiguous["shock_probability"],
                            protection_service_proxy=float(served@scenario.probabilities/total_demand) if total_demand else None))
                        static_rows.append(dict(origin_date=origin,method_id=method,alpha=alpha,**{k:v for k,v in result.items() if k not in {"quantity","matrix"}},matrix=result.get("matrix")))
                    for limit in rconfig["risk_limits"]:
                        result=optimize(c,r,scenario,stock,method="ceiling",alpha=alpha,risk_limit=limit,time_limit=rconfig["solver_seconds"])
                        row=dict(origin_date=origin,method_id="ceiling",alpha=alpha,risk_limit=limit,status=result["status"],gap=result["mip_gap"],objective=result["objective"])
                        if result["quantity"] is not None:
                            row.update(**tail(result["losses"],scenario.probabilities,alpha),max_regret=float((np.asarray(result["losses"])-oracle).max()),
                                       ordered_units=int(sum(result["quantity"])),quantities=json.dumps(result["quantity"].tolist()),fairness="none")
                            fair=fair_slacks(c,result["quantity"],Fairness())
                            served=np.minimum(scenario.demand,stock[:,None]+result["quantity"][:,None]).sum(axis=0)
                            total_demand=scenario.demand.sum(axis=0)@scenario.probabilities
                            row.update(allocation_disparity=fair["disparity"],store_shares=json.dumps(fair["shares"]),
                                       protection_service_proxy=float(served@scenario.probabilities/total_demand) if total_demand else None)
                            frontier_rows.append(row)
                        static_rows.append({**row,"runtime":result["runtime"],"oracle_solves":result.get("oracle_solves",[])})
                # Hard-fairness actual origin feasibility diagnostic, with no relaxation.
                for fair in [Fairness(minimum_share=.2),Fairness(maximum_disparity=.1)]:
                    result=optimize(c,r,scenario,stock,fairness=fair,time_limit=rconfig["solver_seconds"])
                    static_rows.append(dict(origin_date=origin,method_id="hard_fairness",alpha=.75,runtime=result["runtime"],status=result["status"],
                                            objective=result["objective"],gap=result["mip_gap"],fairness=str(fair),
                                            shares=None if result["quantity"] is None else fair_slacks(c,result["quantity"],fair)))
                # At E all risk allocations remain identical after each future mutation.
                for method in METHODS:
                    baseline=optimize(c,r,scenario,stock,**method_options(method),time_limit=rconfig["solver_seconds"])
                    for perturbation in ["baseline","mutate_future_sales","remove_future_sales","mutate_future_prices"]:
                        if perturbation=="baseline":q=baseline["quantity"]
                        else:
                            changed={k:mutations[k][perturbation] for k in series_keys}
                            ct=targets_for("balanced",changed)
                            cs=planning(np.stack([changed[k][0]["historical"] for k in series_keys]))
                            cc,rr,ss=origin_problem(metadata,ct,settings_base)
                            other=optimize(cc,rr,cs,ss,**method_options(method),time_limit=rconfig["solver_seconds"])
                            np.testing.assert_array_equal(baseline["quantity"],other["quantity"]);q=other["quantity"]
                        for m,quantity in zip(metadata.itertuples(),q):
                            snapshot_rows.append(dict(origin_date=origin,method_id=method,series_id=m.series_id,perturbation=perturbation,approved=int(quantity),passed=True))
                observed=np.stack([targets(label_groups[k],origin).actual.to_numpy(int) for k in series_keys])
                # Independent prefix-only evaluation paths with disjoint keyed streams.
                evaluation=[]
                for k in series_keys:
                    b,_,_=bootstrap(histories[k[:2]],origin,samples=8,seed=seed_for(42,"risk-evaluation",*k))
                    evaluation.append(b)
                evaluation=np.stack(evaluation)
                if len({hashlib_array(evaluation[:,j,:]) for j in range(8)})!=8:
                    raise AssertionError("Evaluation worlds must be genuinely distinct joint paths")
                cases=[dict(study="main",family=f,cost="balanced",budget=.7,world_kind="observed_sales",replication=0,
                            actual=shifted(observed,f),fairness=Fairness(),pack=1,moq=0) for f in FAMILIES]
                for family in rconfig["replicated_families"]:
                    for j in range(8):
                        cases.append(dict(study="replicated",family=family,cost="balanced",budget=.7,world_kind="prefix_bootstrap",replication=j,
                                          actual=shifted(evaluation[:,j,:],family),fairness=Fairness(),pack=1,moq=0))
                for study,cost,budget,fair,pack,moq in [
                    ("shortage_scarce","shortage_expensive",.35,Fairness(),1,0),
                    ("overage_ample","overage_expensive",1.4,Fairness(),1,0),
                    ("soft_minimum","balanced",.7,Fairness(minimum_share=.2,soft=True),1,0),
                    ("soft_disparity","balanced",.7,Fairness(maximum_disparity=.1,soft=True),1,0),
                    ("packs","balanced",.7,Fairness(),6,12)]:
                    cases.append(dict(study=study,family="baseline",cost=cost,budget=budget,world_kind="observed_sales",replication=0,
                                      actual=observed.copy(),fairness=fair,pack=pack,moq=moq))
                for case_index,case in enumerate(cases):
                    family=case["family"];scenarios=planning_sets["shifted_probability" if family=="shifted_probability" else "baseline"]
                    cfg_cost=next(x for x in cfg18["costs"] if x["name"]==case["cost"])
                    econ=Economics(**{k:v for k,v in cfg_cost.items() if k!="name"})
                    settings_case={**settings_base,**ops[case["cost"]]}
                    targs=targets_for(case["cost"])
                    lead=dict(name="fixed_14" if family=="long_lead" else "fixed_7",days=[14 if family=="long_lead" else 7],probabilities=[1.])
                    closure=Closure() if family in {"supplier_closure","joint"} else None
                    caseid=f"{case['study']}:{family}:{case['cost']}:{case['budget']}:{case['world_kind']}:{case['replication']}"
                    demand_hash=hashlib_array(case["actual"])
                    eval_hashes.append(dict(origin_date=origin,scenario_config_id=caseid,demand_path_id=demand_hash,world_kind=case["world_kind"],
                                            replication_id=case["replication"],family=family,available_to_optimizer=False))
                    for i,m in enumerate(metadata.itertuples()):
                        supplier=Supplier(tuple(lead["days"]),(1.,),42,f"optimization|{m.series_id}|{origin}")
                        for day in range(29):
                            lag,key,u=supplier.at(day,origin+pd.Timedelta(days=day))
                            shock_rows.append(dict(origin_date=origin,scenario_config_id=caseid,series_id=m.series_id,day_index=day,
                                supplier_shock_id=key,lead_days=lag,uniform=u,delivery_delay=closure(day,None) if closure else 0,demand_path_id=demand_hash))
                    for method in METHODS:
                        meta=dict(origin_date=origin,scenario_config_id=caseid,allocation_group="SELECTED_COHORT_DC",method_id=method,method=method,
                                  study=case["study"],scenario_family=family,world_kind=case["world_kind"],replication_id=case["replication"],
                                  demand_path_id=demand_hash,supplier_shock_id=f"{origin.date()}:{family}:calendar_prekeyed",risk_level=.75,
                                  risk_penalty=float(method.split("_")[1]) if method.startswith("cvar_") else 0.,resource_config_id=f"b{case['budget']}_supplier.75_flow.8",
                                  cost_regime=case["cost"],budget_factor=case["budget"],fairness_configuration=case["study"] if case["fairness"].soft else "none",
                                  forecast_availability_cutoff=origin,model_provenance="frozen historical-block target; retained S15-17 provenance in summary",
                                  scenario_status=STATUS)
                        def allocator(candidates,limits,fairness,stock,unused,economics,day,record=True):
                            result=optimize(candidates,limits,scenarios,stock,**method_options(method),fairness=fairness,
                                         cu=economics.cu,co=economics.co,time_limit=rconfig["solver_seconds"])
                            if result["quantity"] is None:
                                (root/"data/risk").mkdir(exist_ok=True)
                                dump(root/"data/risk/failure_case.json",dict(candidates=asdict(candidates),resources=asdict(limits),
                                    fairness=asdict(fairness),stock=stock,demand=scenarios.demand,probabilities=scenarios.probabilities,
                                    ids=scenarios.ids,options=method_options(method),cu=economics.cu,co=economics.co,result=result))
                                raise AssertionError(f"No risk allocation {origin.date()} {caseid} {method} day {day}: {result}")
                            if record:
                                solver_rows.append({**meta,"decision_day":day,**{k:v for k,v in result.items() if k not in {"quantity","losses","matrix","scores","oracle","regrets","excess","oracle_solves","residuals","raw_residuals","solver_attempts"}},
                                    "residuals":json.dumps(result.get("residuals")),
                                    "raw_residuals":json.dumps(result.get("raw_residuals")),
                                    "solver_attempts":json.dumps(result.get("solver_attempts",[])),
                                    "activation":json.dumps(result.get("activation")),"excess":json.dumps(result.get("excess")),
                                    "oracle_solves":json.dumps(result.get("oracle_solves",[])),"scores":json.dumps(result.get("scores"))})
                                if result.get("matrix"):matrices.append({**meta,"decision_day":day,**result["matrix"]})
                            if result["quantity"] is not None:
                                # Independently recover per-scenario losses from issued q.
                                scored=losses(result["quantity"],stock,scenarios,economics.cu,economics.co)
                                np.testing.assert_allclose(scored,result["losses"],atol=1e-6)
                                if record:
                                    for j,sid in enumerate(scenarios.ids):
                                        loss_rows.append({**meta,"decision_day":day,"planning_scenario_id":sid,"probability":scenarios.probabilities[j],
                                                          "mismatch_cost":scored[j],"preference_cost":result["preference_cost"],
                                                          "oracle_cost":result["oracle"][j] if result.get("oracle") is not None else None,
                                                          "regret":result["regrets"][j] if result.get("regrets") is not None else None})
                                    proof_rows.append(dict(origin_date=origin,scenario_config_id=caseid,method_id=method,decision_day=day,
                                        keys=list(candidates.keys),request=candidates.request.tolist(),pack=candidates.pack.tolist(),moq=candidates.moq.tolist(),
                                        quantity=result["quantity"].tolist(),stock=stock.tolist(),demand=scenarios.demand.tolist(),probabilities=scenarios.probabilities.tolist(),
                                        cu=economics.cu,co=economics.co,alpha=.75,losses=scored.tolist(),scores=result["scores"],preference_cost=result["preference_cost"]))
                            return result
                        kwargs=dict(method="stochastic_milp",costs=econ,lead=lead,budget_factor=case["budget"],supplier_factor=.75,
                                    flow_factor=.8,pack=case["pack"],moq=case["moq"],fairness=case["fairness"],delivery_schedule=closure)
                        result=coordinated(case["actual"],origin,metadata,targs,scenarios.demand,settings_case,
                                           allocator=allocator,**kwargs)
                        for name in ["daily","events","allocations","resources"]:
                            frame=result[name].copy()
                            if "supplier_shock_id" in frame:
                                frame["realized_supplier_shock_id"]=frame.supplier_shock_id
                            frame=frame.assign(**meta)
                            if name=="events":
                                if "previous_arrival_day" not in frame:frame["previous_arrival_day"]=np.nan
                                frame["previous_arrival_day"]=frame.pop("previous_arrival_day").astype(float)
                            sink.write(name,frame)
                        for series,g in result["daily"].groupby("series_id",sort=False):
                            m=metadata[metadata.series_id.eq(series)].iloc[0]
                            run_rows.append({**meta,"series_id":series,"store_id":m.store_id,"item_id":m.item_id,"cohort":m.cohort,
                                             **run_metrics(g),"status":STATUS})
                        if case["study"]=="main" and family=="baseline":
                            # Every live method, all 25 series: future demand/supply prefix gates.
                            def no_record(*args): return allocator(*args,record=False)
                            for kind in ["demand","supplier","hidden_unmet"]:
                                altered=case["actual"].copy()
                                if kind in {"demand","hidden_unmet"}:altered[0,9]+=100000 if kind=="hidden_unmet" else 10000
                                kw=dict(kwargs)
                                if kind=="supplier":kw["delivery_schedule"]=Closure(10,16)
                                other=coordinated(altered,origin,metadata,targs,scenarios.demand,settings_case,allocator=no_record,**kw)
                                cutoff_day=10
                                for name,cols in [("daily",["series_id","day_index","opening_stock","opening_on_order","arrivals","ending_stock","ordered","ending_on_order"]),
                                                  ("allocations",["series_id","day_index","candidate_quantity","approved","position"]),
                                                  ("resources",["day_index","budget_before","committed","budget_after","supplier_remaining"])]:
                                    a=result[name].loc[result[name].day_index.lt(cutoff_day),cols].reset_index(drop=True)
                                    b=other[name].loc[other[name].day_index.lt(cutoff_day),cols].reset_index(drop=True)
                                    pd.testing.assert_frame_equal(a,b)
                                if kind=="hidden_unmet":
                                    second=altered.copy();second[0,9]+=100000
                                    third=coordinated(second,origin,metadata,targs,scenarios.demand,settings_case,allocator=no_record,**kwargs)
                                    a=other["allocations"].loc[other["allocations"].day_index.eq(10),["series_id","approved","position"]]
                                    b=third["allocations"].loc[third["allocations"].day_index.eq(10),["series_id","approved","position"]]
                                    pd.testing.assert_frame_equal(a,b)
                                temporal.append(dict(origin_date=origin,method_id=method,shock_day=10,check=kind,passed=True))
                    if case_index%4==0:print(f"Risk simulation {origin.date()}: case {case_index+1}/{len(cases)}; all shared/queue constraints pass",flush=True)
        finally:sink.close()
        runs=pd.DataFrame(run_rows);pools,operating_risk,paired=summarize_operating(runs)
        _,store_risk,_=summarize_operating(runs.assign(cohort="store:"+runs.store_id))
        front=pd.DataFrame(frontier_rows)
        front["empirically_nondominated"]=False
        for _,g in front.groupby(["origin_date","alpha"]):front.loc[g.index,"empirically_nondominated"]=frontier(g)
        tables=dict(runs=runs,pool_operating=pools,operating_risk=operating_risk,paired_operating=paired,
                    risk_frontier=front,scenario_manifest=pd.DataFrame(manifests),origin_allocations=pd.DataFrame(snapshot_rows),
                    future_invariance=pd.DataFrame(origin_checks),temporal_invariance=pd.DataFrame(temporal),
                    solver_diagnostics=pd.DataFrame(solver_rows),scenario_losses=pd.DataFrame(loss_rows),shock_ledger=pd.DataFrame(shock_rows),
                    evaluation_manifest=pd.DataFrame(eval_hashes),cohort=metadata,store_operating_risk=store_risk,
                    execution_log=pd.DataFrame(executions))
        # Nested solver columns are JSON, preserving variable/status records without Arrow unions.
        diag=tables["solver_diagnostics"]
        for col in ["activation"]:
            if col in diag:diag[col]=diag[col].map(lambda x:json.dumps(x) if not isinstance(x,str) else x)
        for name,frame in tables.items():frame.to_parquet(tmp/"tables"/(name+".parquet"),index=False)
        dump(tmp/"static_solves.json",static_rows);dump(tmp/"solver_matrices.json",matrices)
        dump(tmp/"decision_proofs.json",proof_rows);dump(tmp/"uncertainty_boxes.json",boxes)
        dump(tmp/"config.json",config)
        # A diagnostic copy prevents final reporting/audit failures from losing
        # completed simulations. It is explicitly unverified and never used as
        # a successful published result or as an automatic resume input.
        dump(tmp/"execution_context.json",dict(source_fingerprints=before,warehouse_totals=initial,provenance=provenance,
            origin_mutations=len(origin_checks),temporal_checks=len(temporal),runtime_before_audit=perf_counter()-started))
        checkpoint=(root/"data/risk/unverified_computation").resolve()
        if not checkpoint.is_relative_to((root/"data/risk").resolve()):raise ValueError("Unsafe diagnostic copy path")
        shutil.copytree(tmp,checkpoint,dirs_exist_ok=True)
        from .verification import verify_outputs
        try:
            audit=verify_outputs(tmp)
        except Exception:
            # Preserve completed, explicitly UNVERIFIED computation for diagnosis;
            # a failed audit never replaces the published successful output.
            pending=(root/"data/risk/failed_audit").resolve()
            if not pending.is_relative_to((root/"data/risk").resolve()):raise ValueError("Unsafe audit diagnostic path")
            shutil.copytree(tmp,pending,dirs_exist_ok=True)
            raise
        after={p.relative_to(root).as_posix():fingerprint(p) for p in files}
        if before!=after:raise AssertionError("Source files changed")
        for p,fp in provenance["artifact_hashes"].items():
            if fingerprint(root/p)!=fp:raise AssertionError("Prior model/artifact changed: "+p)
        with connection(warehouse,env) as con:
            if totals(con)!=initial:raise AssertionError("Warehouse totals changed")
        statuses=diag.status.value_counts().to_dict()
        all_static=[s["status"] for s in static_rows]
        oracle_records=[row for s in solver_rows for row in json.loads(s["oracle_solves"])]
        summary=dict(status="passed",scenario_status=STATUS,source_fingerprints=before,warehouse_totals=initial,
            inputs_unchanged=True,provenance=provenance,config=config,cohort_counts=metadata.cohort.value_counts().to_dict(),
            observed_series_origin_paths=75,observed_target_dates=2100,planning_scenarios=8,independent_worlds_per_origin_family=8,
            pool_runs=len(runs)//25,series_runs=len(runs),daily_rows=sink.rows["daily"],
            simulation_replays=len(temporal)+18,origin_mutations=len(origin_checks),origin_allocation_rows=len(snapshot_rows),temporal_checks=len(temporal),
            solver_statuses=statuses,static_statuses=pd.Series(all_static).value_counts().to_dict(),
            oracle_statuses=pd.Series([o["status"] for o in oracle_records]).value_counts().to_dict(),
            max_mip_gap=float(diag.mip_gap.max()),recorded_live_solves=len(diag),recorded_oracle_solves=len(oracle_records),
            presolve_recoveries=sum(e["kind"]=="recovered_attempt" for e in executions),
            static_solves=len(static_rows),verification=audit,table_rows={p.stem:pq.ParquetFile(p).metadata.num_rows for p in (tmp/"tables").glob("*.parquet")},
            total_milp_executions=len(executions),all_execution_statuses=pd.DataFrame(executions).status.value_counts().to_dict(),
            runtime_seconds=perf_counter()-started,peak_memory_mib=peak_memory(),versions={k:version(k) for k in ["scipy","numpy","pandas","duckdb","pyarrow"]},
            unavailable_comparisons="Infeasible epsilon ceilings have no allocation/outcome and are retained as INFEASIBLE; no silent relaxation",
            cost_conventions=dict(planning="Cu/Co forecast mismatch; soft fairness preference separate; no procurement overlay",
                operating="procurement+fixed fee+holding+lost-sales/backlog penalties-terminal on-hand credit; terminal credit=0; initial pipeline prepaid; no resale or lost revenue added",
                real_business_cost="unavailable; M5 sales are censored proxies, not verified demand"))
        dump(tmp/"summary.json",summary)
        from .report import write_report
        write_report(summary,tables,tmp/"risk_report.md")
        publish(tmp,target)
    return summary,target


def hashlib_array(a):
    import hashlib
    return hashlib.sha256(np.asarray(a,dtype="<i8").tobytes()).hexdigest()


def dump(path, value):
    def arrays(v):
        if isinstance(v,np.ndarray):return arrays(v.tolist())
        if isinstance(v,dict):return {k:arrays(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [arrays(x) for x in v]
        return v
    Path(path).write_text(json.dumps(json_ready(arrays(value)),indent=2,default=str,allow_nan=False)+"\n",encoding="utf-8")
