"""Track B: paired complete-policy interventions, never independent daily OPE."""
import numpy as np
import pandas as pd
from demandos.inventory.experiment import ORIGINS
from demandos.simulation.pipeline import Sink
from demandos.simulation.metrics import run_metrics
from demandos.risk.scenarios import shifted,Closure
from demandos.risk.metrics import tail
from .bandit import build_worlds
from .simulation import intervention,prefix_equal,STATUS


def allocation_disparity(allocations,eligible):
    frame=allocations[allocations.series_id.isin(eligible)]
    values=[]
    for _,day in frame.groupby("day_index"):
        groups=day.groupby("store_id")[["candidate_quantity","approved"]].sum()
        groups=groups[groups.candidate_quantity.gt(0)]
        if len(groups):
            share=groups.approved/groups.candidate_quantity
            values.append(float(share.max()-share.min()))
    return float(np.mean(values)) if values else 0.


def run(data,config,directory):
    worlds=build_worlds(data,"trajectory",config["worlds"]);sink=Sink(directory)
    runs=[];solver=[];checks=[];pool_runs=0;replays=0
    try:
        for origin in ORIGINS:
            settings={**data.settings,"resource_targets":data.targets[origin]}
            for family in ["baseline","joint"]:
                cases=[("observed_sales_proxy",-1,data.observed[origin])]+[("prefix_bootstrap",j,worlds[origin][:,j,:]) for j in range(config["worlds"])]
                for kind,world,y in cases:
                    actual=shifted(y,family)
                    for action,policy in enumerate(["mean","cvar_2"]):
                        result=intervention(actual,origin,data.metadata,data.targets[origin],data.scenarios[origin],settings,
                                            action=action,closure=family=="joint",seconds=config["solver_seconds"])
                        meta=dict(origin_date=origin,world_kind=kind,world_id=world,scenario_family=family,policy_id=policy,
                                  resource_pool_id="SELECTED_COHORT_DC",scenario_status=STATUS)
                        for name in ["daily","events","allocations","resources"]:
                            frame=result[name].assign(**meta)
                            if name=="events":
                                if "previous_arrival_day" not in frame:frame["previous_arrival_day"]=np.nan
                                frame["previous_arrival_day"]=frame.pop("previous_arrival_day").astype(float)
                            sink.write("trajectory_"+name,frame)
                        solver.append(result["solver"].assign(**meta))
                        for cohort in ["ALL","continuity","sparse_stress"]:
                            selected=data.metadata if cohort=="ALL" else data.metadata[data.metadata.cohort.eq(cohort)]
                            daily=result["daily"][result["daily"].series_id.isin(selected.series_id)]
                            event=result["events"][result["events"].series_id.isin(selected.series_id)]
                            delay=event[event.event.eq("retimed")]
                            delay_days=float(((delay.arrival_day-delay.previous_arrival_day)*delay.quantity).sum()) if len(delay) else 0.
                            d=daily.demand_proxy.sum();served=daily.immediate_served.sum()
                            runs.append({**meta,"cohort":cohort,"series":len(selected),"operating_cost":float(daily.operational_cost.sum()),
                                "reward":-float(daily.operational_cost.sum()),"demand_units":int(d),"served_units":int(served),
                                "fill":float(served/d) if d else None,"lost_units":int(daily.lost_units.sum()),
                                "delay_order_events":len(delay),"delay_unit_days":delay_days,
                                "allocation_disparity":allocation_disparity(result["allocations"],selected.series_id)})
                        pool_runs+=1
                        if kind=="observed_sales_proxy" and family=="baseline":
                            reset=intervention(actual,origin,data.metadata,data.targets[origin],data.scenarios[origin],settings,action=action)
                            pd.testing.assert_frame_equal(result["daily"],reset["daily"])
                            checks.append({**meta,"check":"full_reset_identical","passed":True});replays+=1
                            for day in [10,20]:
                                altered=actual.copy();altered[0,day-1]+=100000
                                other=intervention(altered,origin,data.metadata,data.targets[origin],data.scenarios[origin],settings,action=action)
                                prefix_equal(result,other,day);replays+=1
                                checks.append({**meta,"check":f"future_demand_day_{day}","passed":True})
                            supply=intervention(actual,origin,data.metadata,data.targets[origin],data.scenarios[origin],settings,action=action,closure=Closure(10,16))
                            prefix_equal(result,supply,10);replays+=1
                            checks.append({**meta,"check":"future_supplier_day_10","passed":True})
                            first=actual.copy();first[0,9]+=100000;second=first.copy();second[0,9]+=100000
                            hidden1=intervention(first,origin,data.metadata,data.targets[origin],data.scenarios[origin],settings,action=action)
                            hidden2=intervention(second,origin,data.metadata,data.targets[origin],data.scenarios[origin],settings,action=action);replays+=2
                            pd.testing.assert_frame_equal(hidden1["allocations"],hidden2["allocations"])
                            checks.append({**meta,"check":"hidden_unmet_not_observable","passed":True})
                print(f"Complete 28-day paired interventions {origin.date()} {family}; reset and temporal gates pass",flush=True)
    finally:sink.close()
    return pd.DataFrame(runs),pd.concat(solver,ignore_index=True),pd.DataFrame(checks),dict(pool_runs=pool_runs,replays=replays,table_rows=sink.rows)


def compare(runs,bootstrap_samples=200):
    keys=["origin_date","world_kind","world_id","scenario_family","cohort","resource_pool_id"]
    mean=runs[runs.policy_id.eq("mean")];risk=runs[runs.policy_id.eq("cvar_2")]
    paired=mean.merge(risk,on=keys,suffixes=("_mean","_cvar"),validate="one_to_one")
    if len(paired)*2!=len(runs):raise AssertionError("Incomplete paired trajectories")
    metrics=["operating_cost","fill","lost_units","delay_unit_days","allocation_disparity"]
    for metric in metrics:paired[metric+"_difference"]=paired[metric+"_cvar"]-paired[metric+"_mean"]
    paired["reward_ate"]=-paired.operating_cost_difference
    risk_rows=[]
    for key,g in runs[runs.world_kind.eq("prefix_bootstrap")].groupby(["origin_date","scenario_family","policy_id","cohort"]):
        if len(g)!=8 or g.world_id.nunique()!=8:raise AssertionError("Eight distinct complete worlds required for operating tail scores")
        risk_rows.append(dict(zip(["origin_date","scenario_family","policy_id","cohort"],key),**tail(g.operating_cost,np.full(8,.125),.75),
                              fill=float(g.served_units.sum()/g.demand_units.sum()) if g.demand_units.sum() else None))
    intervals=[];rng=np.random.default_rng(42)
    for key,g in paired.groupby(["world_kind","scenario_family","cohort"]):
        kind,family,cohort=key
        for metric in metrics:
            name=metric+"_difference";estimate=float(g[name].mean())
            if kind=="prefix_bootstrap":
                blocks=g.groupby("world_id")[name].mean().to_numpy(float)
                if len(blocks)!=8:raise AssertionError("Incomplete world panels across origins")
                draws=np.mean(blocks[rng.integers(0,8,(bootstrap_samples,8))],axis=1)
                lower,upper=np.quantile(draws,[.025,.975]);status="HEURISTIC_8_WORLD_PANELS; all origins paired in block; no real-world coverage"
            else:lower=upper=None;status="UNAVAILABLE_REAL_CI; one historical resource pool, three dependent retrospective origins"
            intervals.append(dict(world_kind=kind,scenario_family=family,cohort=cohort,metric=metric,estimate=estimate,lower=lower,upper=upper,
                                  clusters=8 if kind=="prefix_bootstrap" else 1,origin_world_pairs=len(g),status=status))
    return dict(trajectory_paired=paired,trajectory_operating_risk=pd.DataFrame(risk_rows),trajectory_intervals=pd.DataFrame(intervals))
