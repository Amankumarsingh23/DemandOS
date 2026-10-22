"""Origin-issued SIMULATED vectors; later controller outcomes are separate ledgers."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd

from demandos.inventory.scenarios import bootstrap, seed_for
from demandos.inventory.core import Economics
from demandos.optimization.model import verify, Candidates, Resources
from demandos.risk.pipeline import origin_problem
from demandos.risk.model import solve, Scenarios
from demandos.risk.metrics import tail
from demandos.causal.simulation import intervention, prefix_equal
from .core import require, Blocked, digest, canonical, STATUS
from .features import KEY, eligible_history

POLICY = "origin-p50-14d-target-mean-mismatch-v1"


def frozen_inputs(snapshot,metadata,origin,settings):
    history=eligible_history(snapshot,origin);targets=[];paths=[]
    for m in metadata.itertuples():
        h=history[history.store_id.eq(m.store_id)&history.item_id.eq(m.item_id)]
        recent=h[h.date.gt(pd.Timestamp(origin)-pd.Timedelta(days=56))]
        targets.append(int(np.ceil(14*recent.units_sold.mean())))
        paths.append(bootstrap(h,origin,samples=8,lookback=56,block=7,horizon=28,seed=seed_for(42,"serving-plan",origin,m.series_id))[0])
    scenarios=np.stack(paths)
    _,resources,_=origin_problem(metadata,np.array(targets),settings,budget=.7,supplier=.75,flow=.8)
    resources=Resources(resources.budget,resources.flow_volume,dict(sorted(resources.suppliers.items())),dict(sorted(resources.receiving.items())))
    protection=scenarios[:,:,:14].sum(axis=2).astype(float)
    return np.array(targets,int),Scenarios(protection,np.full(8,.125),tuple(f"history_{i}" for i in range(8))),resources,scenarios


def target_vector(forecasts,metadata):
    rows=forecasts[forecasts.horizon.le(14)].groupby(KEY).q50.sum()
    require(len(rows)==len(metadata),"DECISION_FORECAST_COVERAGE")
    return np.array([int(np.ceil(rows.loc[(r.store_id,r.item_id)])) for r in metadata.itertuples()],int)


def resource_record(resources):
    return dict(budget=resources.budget,flow_volume=resources.flow_volume,suppliers=resources.suppliers,receiving=resources.receiving)


def plan(forecasts,metadata,settings,scenarios,resources,*,force_infeasible=False):
    target=target_vector(forecasts,metadata)
    c,_,stock=origin_problem(metadata,target,settings)
    c=replace(c,pack=np.full(len(c.keys),settings.get("pack",1)),moq=np.full(len(c.keys),settings.get("moq",0)))
    if force_infeasible:
        c=replace(c,request=np.maximum(c.request,1),minimum=np.ones(len(c.keys)))
        resources=Resources(0,0,{s:0 for s in c.suppliers},{s:0 for s in c.stores})
    result=solve(c,resources,scenarios,stock,method="mean",cu=40.,co=40.,time_limit=settings["solver_seconds"])
    require(result["quantity"] is not None and result["status"] in {"OPTIMAL","FEASIBLE_NOT_PROVEN_OPTIMAL"},"OPTIMIZER_INFEASIBLE",result["status"])
    verify(c,resources,result["quantity"])
    resource=resource_record(resources);rows=[]
    for i,m in enumerate(metadata.itertuples()):
        rows.append(dict(store_id=m.store_id,item_id=m.item_id,series_id=m.series_id,origin_date=pd.Timestamp(forecasts.origin_date.iloc[0]),
            approved_quantity=int(result["quantity"][i]),request=int(c.request[i]),target_units=int(target[i]),purchase=float(c.purchase[i]),fee=float(c.fee[i]),
            volume=float(c.volume[i]),supplier=c.suppliers[i],pack=int(c.pack[i]),moq=int(c.moq[i]),
            policy_version=POLICY,optimizer_status=result["status"],optimizer_gap=float(result["mip_gap"] or 0),
            resource_digest=digest(resource),resource_limits=canonical(resource).decode(),
            model_version=forecasts.model_version.iloc[0],model_digest=forecasts.model_digest.iloc[0],
            feature_contract_digest=forecasts.feature_contract_digest.iloc[0],
            prediction_ids=json.dumps(forecasts.loc[forecasts.store_id.eq(m.store_id)&forecasts.item_id.eq(m.item_id),"prediction_id"].tolist()),
            validation_status="INDEPENDENT_RESOURCE_CHECK_PASSED",state_status="SYNTHETIC_OPENING_STOCK_10_PIPELINE_3",scenario_status=STATUS))
    frame=pd.DataFrame(rows);validate_decisions(frame)
    return frame,dict(status=result["status"],gap=result["mip_gap"],runtime=result["runtime"],resource=resource,policy=POLICY,
                       resource_use=verify(c,resources,result["quantity"])[0])


def validate_decisions(frame):
    if frame.empty:return
    fields={*KEY,"series_id","origin_date","approved_quantity","request","purchase","fee","volume","supplier","pack","moq","resource_limits","resource_digest"}
    require(fields.issubset(frame) and not frame.duplicated([*KEY,"origin_date"]).any(),"DECISION_KEYS")
    resources=json.loads(frame.resource_limits.iloc[0]);require(frame.resource_limits.eq(frame.resource_limits.iloc[0]).all() and frame.resource_digest.eq(digest(resources)).all(),"RESOURCE_LINEAGE")
    c=Candidates(tuple(frame.series_id),frame.request.to_numpy(),frame.purchase.to_numpy(),frame.volume.to_numpy(),np.ones(len(frame)),
                 tuple(frame.supplier),tuple(frame.store_id),frame.fee.to_numpy(),frame.pack.to_numpy(),frame.moq.to_numpy(),np.zeros(len(frame)))
    try:verify(c,Resources(**resources),frame.approved_quantity.to_numpy())
    except (AssertionError,ValueError) as exc:raise Blocked("RESOURCE_FEASIBILITY",exc) from exc


def audit_simulation(result):
    d=result["daily"];a=result["allocations"];r=result["resources"];e=result["events"]
    require((d.opening_stock+d.arrivals==d.immediate_served+d.old_backlog_fulfilled+d.ending_stock).all(),"INVENTORY_CONSERVATION")
    require((d.demand_proxy==d.immediate_served+d.lost_units+d.new_backlog).all(),"DEMAND_CONSERVATION")
    require((d.opening_on_order-d.arrivals+d.ordered==d.ending_on_order).all(),"QUEUE_CONSERVATION")
    require(d[["ending_stock","ending_on_order","ending_backlog"]].ge(0).all().all(),"NEGATIVE_INVENTORY")
    np.testing.assert_allclose(d.operational_cost,.2*d.ending_stock+40*d.lost_units+2*d.ending_backlog+60*d.ordered+5*d.ordered.gt(0)-d.terminal_credit,atol=1e-7)
    order=d[["series_id","day_index","ordered"]].merge(a[["series_id","day_index","approved"]],on=["series_id","day_index"],validate="one_to_one")
    require((order.ordered==order.approved).all() and a.approved.le(a.candidate_quantity).all() and a.approved.ge(0).all() and a.approved.eq(np.floor(a.approved)).all(),"SIMULATED_APPROVAL_FEASIBILITY")
    require((a.approved%a.pack==0).all() and (a.approved.eq(0)|a.approved.ge(a.moq)).all(),"SIMULATED_PACK_MOQ")
    for _,g in d.groupby("series_id"):
        require(len(g)==28 and g.day_index.nunique()==28,"SIMULATED_HORIZON_COVERAGE")
    previous=None;supplier=None
    for row in r.itertuples():
        require(row.budget_after>=-1e-7 and row.flow_used<=row.flow_limit+1e-7,"SIMULATED_RESOURCE_VIOLATION")
        require(abs(row.budget_before-row.committed-row.budget_after)<1e-7 and (previous is None or abs(previous-row.budget_before)<1e-7),"PERSISTENT_CASH_CONSERVATION")
        day=a[a.day_index.eq(row.day_index)]
        require(abs(day.purchase@day.approved+day.fee@day.activation-row.committed)<1e-7,"PROCUREMENT_LEDGER")
        old=json.loads(row.supplier_initial) if supplier is None else supplier;new=json.loads(row.supplier_remaining)
        use=day.groupby("supplier").approved.sum()
        require(all(v>=-1e-7 and abs(old[k]-use.get(k,0)-v)<1e-7 for k,v in new.items()),"PERSISTENT_SUPPLIER_CONSERVATION")
        for store,limit in json.loads(row.receiving_limit).items():require(day.loc[day.store_id.eq(store),"approved"].sum()<=limit+1e-7,"STORE_QUOTA")
        previous=row.budget_after;supplier=new
    for _,g in e.groupby(["series_id","order_id"]):
        require(g.event.isin(["placed","initial_outstanding"]).sum()==1 and g.event.isin(["received","terminal_outstanding"]).sum()==1 and g.quantity.nunique()==1,"ORDER_LIFECYCLE")
    for series,g in d.groupby("series_id"):
        receipts=e[e.series_id.eq(series)&e.event.eq("received")].groupby("event_day").quantity.sum()
        require(all(row.arrivals==receipts.get(row.day_index,0) for row in g.itertuples()),"RECEIPT_STOCK_RECONCILIATION")
    return dict(cost=float(d.operational_cost.sum()),demand=int(d.demand_proxy.sum()),served=int(d.immediate_served.sum()),
                lost=int(d.lost_units.sum()),fill=float(d.immediate_served.sum()/d.demand_proxy.sum()) if d.demand_proxy.sum() else None,
                budget_used=float(r.committed.sum()),budget_initial=float(r.budget_initial.iloc[0]),
                minimum_stock=int(d.ending_stock.min()),queue_events=len(e),fallbacks=int(result["solver"].get("fallback",pd.Series(dtype=str)).notna().sum()))


def downstream(actual,origin,metadata,forecasts,scenarios,settings,resource_targets):
    targets=target_vector(forecasts,metadata)
    result=intervention(np.asarray(actual,int),origin,metadata,targets,scenarios,{**settings,"resource_targets":resource_targets},action=0)
    metrics=audit_simulation(result)
    return result,metrics
