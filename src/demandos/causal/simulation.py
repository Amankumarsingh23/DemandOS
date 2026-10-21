"""Existing inventory engines; separate single-allocation and full-policy tracks."""
import json
import numpy as np
import pandas as pd
from demandos.inventory.core import Economics
from demandos.inventory.core import counts
from demandos.simulation.engine import simulate,Costs,Supplier
from demandos.simulation.policies import Order,Policy
from demandos.simulation.metrics import run_metrics
from demandos.optimization.coordinator import coordinated
from demandos.optimization.model import verify,Fairness,fair_slacks
from demandos.risk.pipeline import origin_problem
from demandos.risk.model import solve
from demandos.risk.scenarios import Closure

STATUS="SIMULATED_SYNTHETIC_LOGGING_MODEL_WORLD"
ECONOMICS=Economics(100,60,20)
OPERATIONS=Costs(.2,40.,2.,5.,60.)


def catalog(metadata,targets,scenarios,settings,*,stock=10,budget=.7,flow=.8,seconds=3.):
    cfg={**settings,"on_hand":stock}
    candidates,resources,position=origin_problem(metadata,targets,cfg,budget=budget,supplier=.75,flow=flow)
    quantities=[];diagnostics=[]
    for method in ["mean","mean_cvar"]:
        result=solve(candidates,resources,scenarios,position,method=method,penalty=2. if method=="mean_cvar" else 0.,
                     alpha=.75,cu=40.,co=40.,time_limit=seconds)
        if result["quantity"] is None:raise AssertionError("No feasible action in fixed catalogue: "+result["status"])
        verify(candidates,resources,result["quantity"])
        quantities.append(result["quantity"])
        diagnostics.append(dict(method=method,status=result["status"],gap=result["mip_gap"],runtime=result["runtime"],
                                quantity=result["quantity"].tolist(),objective=result["objective"],matrix=result["matrix"],
                                resource_use=verify(candidates,resources,result["quantity"])[0],
                                fairness=fair_slacks(candidates,result["quantity"],Fairness())))
    return np.stack(quantities),candidates,resources,diagnostics


def one_allocation(demand,origin,metadata,quantity,*,stock=10,on_order=3,world_id=0):
    """Issue one whole feasible vector at E, then no further orders for 14 days.

    The original simulator treats day-zero pipeline as prepaid. Add the newly
    issued vector's purchase/activation cash ONCE, keeping existing pipeline
    prepaid. No Season 18 mismatch cost is added. No oracle chooses actions.
    """
    y=np.asarray(demand)
    if y.shape!=(len(metadata),14):raise ValueError("Single-allocation track needs complete 14-day pool demand")
    quantity=counts(quantity)
    if quantity.shape!=(len(metadata),):raise ValueError("Aligned whole-pool integer allocation required")
    ledger=[];events=[];runs=[]
    for i,m in enumerate(metadata.itertuples()):
        supplier=Supplier((7,),(1.,),42,f"causal-one-step|{origin}|{world_id}|{m.series_id}")
        init=[Order("prepaid",0,7,on_order,7,"known_at_origin")] if on_order else []
        if quantity[i]:init.append(Order("chosen_action",0,7,int(quantity[i]),7,"known_at_origin"))
        daily,event=simulate(y[i],origin,Policy("RS",0,0),OPERATIONS,supplier,on_hand=stock,initial_orders=init)
        cash=60.*int(quantity[i])+5.*(quantity[i]>0)
        meta=dict(series_id=m.series_id,store_id=m.store_id,item_id=m.item_id,cohort=m.cohort,origin_date=origin,world_id=world_id,
                  initial_stock=stock,initial_prepaid_pipeline=on_order,initial_action_quantity=int(quantity[i]),decision_cash=cash,scenario_status=STATUS)
        ledger.append(daily.assign(**meta))
        events.append(event.assign(**meta))
        metric=run_metrics(daily)
        runs.append({**meta,**metric,"total_cost":metric["operational_cost"]+cash,"reward":-metric["operational_cost"]-cash})
    return dict(daily=pd.concat(ledger,ignore_index=True),events=pd.concat(events,ignore_index=True),runs=pd.DataFrame(runs))


def intervention(demand,origin,metadata,targets,scenarios,settings,*,action=0,closure=False,seconds=3.):
    """Fresh coordinated engines and ledgers on EVERY call; no state reuse."""
    records=[]
    def allocator(c,r,f,stock,unused,cost,day):
        result=solve(c,r,scenarios,stock,method="mean" if action==0 else "mean_cvar",penalty=0. if action==0 else 2.,alpha=.75,
                     fairness=f,cu=cost.cu,co=cost.co,time_limit=seconds)
        records.append(dict(day_index=day,status=result["status"],gap=result["mip_gap"],runtime=result["runtime"]))
        return result
    result=coordinated(demand,origin,metadata,targets,scenarios.demand,settings,method="stochastic_milp",costs=ECONOMICS,
                       lead=dict(days=[7],probabilities=[1.]),budget_factor=.7,supplier_factor=.75,flow_factor=.8,
                       allocator=allocator,delivery_schedule=Closure() if closure is True else closure if closure else None)
    result["solver"]=pd.DataFrame(records)
    return result


def prefix_equal(first,second,day):
    for name,cols in [
        ("daily",["series_id","day_index","opening_stock","opening_on_order","arrivals","ending_stock","ending_on_order","ordered"]),
        ("allocations",["series_id","day_index","candidate_quantity","approved","position"]),
        ("resources",["day_index","budget_before","committed","budget_after","supplier_remaining"])]:
        a=first[name].loc[first[name].day_index.lt(day),cols].reset_index(drop=True)
        b=second[name].loc[second[name].day_index.lt(day),cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
