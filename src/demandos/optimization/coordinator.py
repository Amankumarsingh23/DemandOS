"""Two-phase shared decisions using the ORIGINAL Season 19 timeline generator."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd

from demandos.simulation.engine import steps, Costs, Supplier, STATUS
from demandos.simulation.policies import Order, Policy, Action
from demandos.inventory.core import Economics
from .model import Candidates, Resources, Fairness, solve, baseline, verify, resource_use, fair_slacks

METHODS=["unconstrained_reference","proportional","greedy","deterministic_milp","stochastic_milp"]


def origin_snapshot(metadata,targets,scenarios,settings,costs):
    """Non-committed day-zero diagnostic: no realized future-demand argument.

    This proves first-stage decisions at E use only initial state and frozen
    scenario information; the live simulation still orders after day demand.
    """
    n=len(metadata);target=np.asarray(targets,int);stock=np.full(n,settings["on_hand"]+settings["on_order"],float)
    req=np.maximum(0,target-stock).astype(int);suppliers=tuple(metadata.dept_id);stores=tuple(metadata.store_id)
    c=Candidates(tuple(metadata.series_id),req,np.full(n,settings["purchase"]),metadata.volume.to_numpy(float),
        costs.cu*np.mean(scenarios>stock[:,None],axis=1),suppliers,stores,np.full(n,settings["order_fee"]),np.ones(n),np.zeros(n),np.zeros(n))
    resources=Resources(.7*(4*settings["purchase"]*target.sum()+4*n*settings["order_fee"]),max(1.,float(c.volume@target)*.8),
        {group:float(4*target[np.array(suppliers)==group].sum()) for group in sorted(set(suppliers))},
        {store:max(1.,float(target[np.array(stores)==store].sum())*settings["receiving_factor"]) for store in sorted(set(stores))})
    result=solve(c,resources,demand=scenarios,probabilities=np.full(scenarios.shape[1],1/scenarios.shape[1]),stock=stock,
        cu=costs.cu,co=costs.co,time_limit=settings["solver_seconds"])
    if result["quantity"] is None:raise AssertionError("Origin snapshot has no valid incumbent")
    verify(c,resources,result["quantity"])
    return result


def coordinated(actual, origin, metadata, targets, scenarios, settings, *, method, costs, lead,
                budget_factor=.7,supplier_factor=1.,flow_factor=.8,pack=1,moq=0,fairness=Fairness(),mode="lost_sales",
                allocator=None,delivery_schedule=None):
    """Candidate forecasts/resources are frozen; only POST-demand observable IP evolves.

    All 25 timelines stop at their observation yield BEFORE allocating. One
    shared first-stage order vector is sent back; scenario recourse is accounting.
    """
    if method not in METHODS:raise ValueError("Unknown allocation method")
    keys=tuple(metadata.series_id);n=len(keys)
    if keys!=tuple(sorted(keys)) or actual.shape!=(n,28) or scenarios.shape[0]!=n:raise ValueError("Aligned sorted 25-series target matrix required")
    volume=metadata.volume.to_numpy(float);suppliers=tuple(metadata.dept_id);stores=tuple(metadata.store_id)
    target=np.asarray(targets,int);prob=np.full(scenarios.shape[1],1/scenarios.shape[1])
    basis=np.asarray(settings.get("resource_targets",target),int)
    if basis.shape!=(n,) or (basis<0).any():raise ValueError("Aligned nonnegative frozen resource reference required")
    budget_initial=budget_factor*(4*settings["purchase"]*basis.sum()+4*n*settings["order_fee"])
    supplier_initial={s:float(4*basis[np.array(suppliers)==s].sum()*supplier_factor) for s in sorted(set(suppliers))}
    flow_limit=max(1.,float(volume@basis)*flow_factor)
    receiving={s:max(1.,float(basis[np.array(stores)==s].sum())*settings["receiving_factor"]) for s in sorted(set(stores))}
    remaining_budget=budget_initial;remaining_supplier=supplier_initial.copy()
    timelines=[];observations=[];finished=[];allocations=[];balances=[];diagnostics=[];matrix_manifest=[]
    for i,key in enumerate(keys):
        supplier=Supplier(tuple(lead["days"]),tuple(lead["probabilities"]),settings["seed"],f"optimization|{key}|{origin}")
        initial_lead,shock,_=supplier.at(0,origin)
        init=(Order("initial_0",0,initial_lead,settings["on_order"],initial_lead,shock),) if settings["on_order"] else ()
        operating=Costs(settings["holding"],settings["lost_penalty"],settings["backlog_daily"],settings["order_fee"],settings["purchase"])
        timeline=steps(actual[i],origin,operating,supplier,on_hand=settings["on_hand"],initial_orders=init,mode=mode,
                       delivery_schedule=delivery_schedule)
        timelines.append(timeline);observations.append(next(timeline))
    for day in range(1,29):
        review=(day-1)%settings["review_days"]==0
        # Frozen S from the declared source; no realized future demand in requests.
        requests=np.array([Policy("RS",0,int(target[i]),review_days=settings["review_days"]).decide(o).quantity for i,o in enumerate(observations)])
        stock=np.array([o.position for o in observations],float)
        # Heuristic benefit: assumed shortage value × origin-scenario exceedance.
        benefit=costs.cu*np.mean(scenarios>stock[:,None],axis=1)
        c=Candidates(keys,requests,np.full(n,settings["purchase"]),volume,benefit,suppliers,stores,
            np.full(n,settings["order_fee"]),np.full(n,pack),np.full(n,moq),np.zeros(n))
        resources=Resources(max(0.,remaining_budget),flow_limit,{s:max(0.,v) for s,v in remaining_supplier.items()},receiving)
        q=np.zeros(n,int);result=dict(status="NO_REQUEST",mip_gap=None,runtime=0.,objective=None)
        lp=None
        if requests.any():
            if method=="unconstrained_reference":
                q=(requests//pack*pack).astype(int);q[q<moq]=0
                result["status"]="UNCONSTRAINED_REFERENCE_NOT_RESOURCE_FEASIBLE"
            elif method in ["proportional","greedy"]:
                q=baseline(c,resources,method);result["status"]="FEASIBLE_BASELINE"
            elif allocator is not None:
                result=allocator(c,resources,fairness,stock,scenarios,costs,day)
                if result.get("quantity") is None:
                    raise ValueError("Risk allocator has no feasible plan; hard limits are never relaxed: "+result["status"])
                q=result["quantity"]
                matrix_manifest.append(dict(day_index=day,method=method,**result.get("matrix",{})))
            else:
                kwargs=dict(demand=scenarios,probabilities=prob,stock=stock,cu=costs.cu,co=costs.co) if method=="stochastic_milp" else {}
                result=solve(c,resources,fairness,time_limit=settings["solver_seconds"],**kwargs)
                if result["quantity"] is None:
                    # The declared fallback may drop ONLY soft preferences.
                    q=np.zeros(n,int);verify(c,resources,q,fairness)
                    result["fallback"]="EXPLICIT_ZERO_ORDER_HARD_FEASIBLE"
                else:q=result["quantity"]
                lp=solve(c,resources,fairness,time_limit=settings["solver_seconds"],relax=True,**kwargs)
                if lp["objective"] is not None and result["objective"] is not None:
                    if lp["objective"]>result["objective"]+1e-5:raise AssertionError("LP bound exceeds minimization integer incumbent")
                matrix_manifest.append(dict(day_index=day,method=method,**result.get("matrix",{})))
            if method!="unconstrained_reference":verify(c,resources,q,fairness)
        use=resource_use(c,q);fair=fair_slacks(c,q,fairness)
        before_budget=remaining_budget
        remaining_budget-=use["budget"]
        for s in remaining_supplier:remaining_supplier[s]-=use["suppliers"].get(s,0.)
        if method!="unconstrained_reference" and (remaining_budget < -1e-7 or min(remaining_supplier.values()) < -1e-7):raise AssertionError("Persistent resource overcommitment")
        if abs(before_budget-use["budget"]-remaining_budget)>1e-7:raise AssertionError("Budget conservation")
        balances.append(dict(day_index=day,budget_initial=budget_initial,budget_before=before_budget,committed=use["budget"],budget_after=remaining_budget,
            flow_limit=flow_limit,flow_used=use["flow_volume"],supplier_initial=json.dumps(supplier_initial,sort_keys=True),
            supplier_remaining=json.dumps(remaining_supplier,sort_keys=True),supplier_used=json.dumps(use["suppliers"],sort_keys=True),
            receiving_limit=json.dumps(receiving,sort_keys=True),receiving_used=json.dumps(use["receiving"],sort_keys=True),
            constrained=method!="unconstrained_reference",status=STATUS))
        if review:
            diagnostics.append(dict(day_index=day,method=method,status=result["status"],gap=result.get("mip_gap"),objective=result.get("objective"),
                runtime=result["runtime"],fallback=result.get("fallback"),lp_status=lp["status"] if lp else None,
                lp_objective=lp.get("objective") if lp else None,
                lp_minimization_marginals=json.dumps(lp.get("minimization_rhs_marginals")) if lp else None,
                integer_gap=result["objective"]-lp["objective"] if lp and lp.get("objective") is not None and result.get("objective") is not None else None,
                fairness_min_slack=sum(fair["minimum_shortfall"].values()),fairness_disparity=fair["disparity"],fairness_disparity_slack=fair["disparity_slack"]))
        for i,observation in enumerate(observations):
            allocations.append(dict(series_id=keys[i],store_id=metadata.store_id.iloc[i],item_id=metadata.item_id.iloc[i],day_index=day,
                simulation_date=pd.Timestamp(observation.simulation_date),order_request_id=f"{keys[i]}:{day}",candidate_quantity=int(requests[i]),approved=int(q[i]),
                activation=int(q[i]>0),position=observation.position,blocked=int(requests[i]-q[i]),benefit=benefit[i],
                objective_contribution=float(costs.cu*np.mean(np.maximum(scenarios[i]-stock[i]-q[i],0))+costs.co*np.mean(np.maximum(stock[i]+q[i]-scenarios[i],0))) if method=="stochastic_milp" else float(benefit[i]*q[i]),
                purchase=c.purchase[i],fee=c.fee[i],volume=c.volume[i],supplier=suppliers[i],warehouse_group="SELECTED_COHORT_DC",
                pack=pack,moq=moq,request_share=float(q[i]/requests[i]) if requests[i] else None,store_share=fair["shares"].get(stores[i]),status=STATUS))
        # Commit simultaneously. Advance every engine only after the vector is fixed.
        next_observations=[]
        for i,timeline in enumerate(timelines):
            try:next_observations.append(timeline.send(Action(int(q[i]),"shared_approval:"+method,0,int(target[i]),observations[i].simulation_date)))
            except StopIteration as done:
                if day!=28:raise AssertionError("Incomplete coordinated path")
                finished.append(done.value)
        observations=next_observations
    ledgers=[];events=[]
    for i,(daily,event) in enumerate(finished):
        meta=dict(series_id=keys[i],store_id=metadata.store_id.iloc[i],item_id=metadata.item_id.iloc[i],cohort=metadata.cohort.iloc[i],origin_date=pd.Timestamp(origin))
        ledgers.append(daily.assign(**meta))
        if not event.empty:events.append(event.assign(**meta))
    return dict(daily=pd.concat(ledgers,ignore_index=True),events=pd.concat(events,ignore_index=True) if events else pd.DataFrame(),
                allocations=pd.DataFrame(allocations),resources=pd.DataFrame(balances),solver=pd.DataFrame(diagnostics),matrices=matrix_manifest)
