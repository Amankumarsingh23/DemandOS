"""Independent persisted-ledger audit: no solver matrices or engine assertions."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from .metrics import tail, epigraph_cvar, frontier


def verify_outputs(directory):
    root=Path(directory);checks={}
    with duckdb.connect() as con:
        con.execute("set threads=2");con.execute("set memory_limit='1GB'")
        for p in (root/"tables").glob("*.parquet"):
            con.read_parquet(str(p)).create_view(p.stem)
        def gate(name,sql):
            count=con.execute(sql).fetchone()[0]
            if count:raise AssertionError(f"Independent risk gate {name}: {count} failures")
            checks[name]=True
        keys="origin_date,scenario_config_id,method_id,series_id"
        gate("unique_daily",f"select count(*) from (select {keys},day_index from daily group by all having count(*)<>1)")
        gate("complete_horizon",f"select count(*) from (select {keys} from daily group by all having count(*)<>28 or min(day_index)<>1 or max(day_index)<>28)")
        gate("calendar","select count(*) from daily where date_diff('day',origin_date,simulation_date)<>day_index")
        gate("stock_backlog_demand","""select count(*) from daily where opening_stock+arrivals<>immediate_served+old_backlog_fulfilled+ending_stock or
            demand_proxy<>immediate_served+lost_units+new_backlog or opening_backlog+new_backlog<>ending_backlog+old_backlog_fulfilled or
            ending_on_order<>on_order_before_action+ordered or least(ending_stock,ending_backlog,ending_on_order,ordered)<0""")
        gate("queue_conservation","""select count(*) from daily where opening_on_order-arrivals+ordered<>ending_on_order""")
        gate("operating_accounting","select count(*) from daily where abs(operational_cost-holding_cost-lost_cost-backlog_cost-ordering_cost-procurement_cost+terminal_credit)>1e-6")
        gate("integer_pack_moq","""select count(*) from allocations where approved<0 or approved>candidate_quantity or approved<>round(approved)
            or approved%pack<>0 or (approved>0 and approved<moq) or activation<>(approved>0)::integer""")
        gate("approved_equals_placed",f"select count(*) from daily d join allocations a using({keys},day_index) where d.ordered<>a.approved or d.position_before_order<>a.position")
        gate("paired_worlds","""select count(*) from (select origin_date,scenario_config_id,series_id,day_index from daily group by all
             having count(distinct method_id)<>6 or count(distinct demand_proxy)<>1 or count(distinct demand_path_id)<>1)""")
        gate("budget_conservation","select count(*) from resources where abs(budget_before-committed-budget_after)>1e-7 or budget_after < -1e-7 or flow_used>flow_limit+1e-7")
        gate("persistent_cash","""select count(*) from (select *,lag(budget_after) over(partition by origin_date,scenario_config_id,method_id order by day_index) previous_budget from resources)
             where (day_index=1 and budget_before<>budget_initial) or (day_index>1 and abs(previous_budget-budget_before)>1e-7)""")
        gate("cash_equals_operating_purchases","""select count(*) from
            (select origin_date,scenario_config_id,method_id,day_index,sum(procurement_cost+ordering_cost) cash from daily group by all) d
            join resources r using(origin_date,scenario_config_id,method_id,day_index) where abs(d.cash-r.committed)>1e-7""")
        gate("origin_nonanticipativity","""select count(*) from (select origin_date,method_id,series_id from origin_allocations group by all
             having count(*)<>4 or count(distinct approved)<>1 or min(approved)<0)""")
        gate("availability_cutoff","select count(*) from scenario_manifest where availability_cutoff>origin_date")
        gate("scenario_probability","""select count(*) from (select origin_date,store_id,item_id,planning_family from scenario_manifest group by all
             having count(*)<>8 or count(distinct scenario_id)<>8 or min(probability)<0 or abs(sum(probability)-1)>1e-12)""")
        gate("order_lifecycle",f"""select count(*) from (select {keys},order_id from events group by all having
            count(*) filter(where event in ('placed','initial_outstanding'))<>1 or count(*) filter(where event in ('received','terminal_outstanding'))<>1
            or count(distinct quantity)<>1 or min(lead_time)<1 or max(abs(arrival_day-issue_day-lead_time))<>0
            or max(case when event='received' then abs(event_day-arrival_day) else 0 end)<>0)""")
        gate("retiming_forward_only","""select count(*) from events where event='retimed' and
             (arrival_day<=previous_arrival_day or event_day<>previous_arrival_day or quantity<=0)""")
        gate("receipt_events_equal_arrivals",f"""select count(*) from daily d full join
            (select {keys},event_day day_index,sum(quantity) units from events where event='received' group by all) e
            using({keys},day_index) where coalesce(d.arrivals,0)<>coalesce(e.units,0)""")
        gate("run_metrics",f"""select count(*) from (select {keys},sum(operational_cost) as total_cost,sum(demand_proxy) as total_demand,sum(immediate_served) as total_served from daily group by all) d
             join runs r using({keys}) where abs(d.total_cost-r.operational_cost)>1e-6 or d.total_demand<>r.demand_units or d.total_served<>r.immediate_served_units""")
        gate("risk_solvers_valid","select count(*) from solver_diagnostics where status not in ('OPTIMAL','FEASIBLE_NOT_PROVEN_OPTIMAL') or (status='OPTIMAL' and mip_gap>1e-7)")
        gate("temporal_gates_passed","select count(*) from temporal_invariance where not passed")
        gate("prefix_gates_passed","select count(*) from future_invariance where not passed")
        gate("regret_accounting","select count(*) from scenario_losses where oracle_cost is not null and (regret < -1e-6 or abs(mismatch_cost+preference_cost-oracle_cost-regret)>1e-6)")
        # Independent aggregation in business units; JSON is only the persisted limits.
        allocation=con.execute("""select origin_date,scenario_config_id,method_id,day_index,
            sum(purchase*approved+fee*activation) cash,sum(volume*approved) flow
            from allocations group by all""").df()
        supplier=con.execute("""select origin_date,scenario_config_id,method_id,day_index,supplier,sum(approved) units from allocations group by all""").df()
        stores=con.execute("""select origin_date,scenario_config_id,method_id,day_index,store_id,sum(approved) units,
            sum(candidate_quantity) requests from allocations group by all""").df()
        resources=con.execute("select * from resources").df()
        count=con.execute("select count(*) from daily").fetchone()[0]
        retimed=con.execute("select count(*) from events where event='retimed'").fetchone()[0]
    group=["origin_date","scenario_config_id","method_id","day_index"]
    use=resources.merge(allocation,on=group,validate="one_to_one")
    np.testing.assert_allclose(use.committed,use.cash,atol=1e-7)
    np.testing.assert_allclose(use.flow_used,use.flow,atol=1e-7)
    suppliers={key:dict(zip(g.supplier,g.units)) for key,g in supplier.groupby(group)}
    store_groups={key:g for key,g in stores.groupby(group)}
    fair_rows=[]
    for _,g in resources.groupby(group[:3]):
        previous=None
        for row in g.sort_values("day_index").itertuples():
            key=tuple(getattr(row,c) for c in group)
            initial=json.loads(row.supplier_initial);remaining=json.loads(row.supplier_remaining)
            before=initial if previous is None else previous
            for name in initial:
                used=suppliers[key].get(name,0)
                if abs(before[name]-used-remaining[name])>1e-7 or remaining[name]<-1e-7:
                    raise AssertionError("Persistent supplier commitment violation")
            previous=remaining
            st=store_groups[key]
            for store,limit in json.loads(row.receiving_limit).items():
                if st.loc[st.store_id.eq(store),"units"].sum()>limit+1e-7:raise AssertionError("Store quota violation")
            if (row.day_index-1)%7==0:
                positive=st[st.requests.gt(0)].copy();positive["share"]=positive.units/positive.requests
                for s in positive.itertuples():
                    fair_rows.append(dict(zip(group,key),store_id=s.store_id,requested=s.requests,approved=s.units,request_share=s.share,
                                          min_share_shortfall=max(0,.2-s.share) if row.study=="soft_minimum" else 0.,
                                          disparity=float(positive.share.max()-positive.share.min()),
                                          disparity_slack=max(0,float(positive.share.max()-positive.share.min())-.1) if row.study=="soft_disparity" else 0.))
    checks["independent_resources_and_store_denominators"]=True
    # Re-score decision losses without the model's loss function or auxiliaries.
    proofs=json.loads((root/"decision_proofs.json").read_text())
    for proof in proofs:
        q=np.asarray(proof["quantity"]);stock=np.asarray(proof["stock"]);d=np.asarray(proof["demand"])
        shortage=np.maximum(d-stock[:,None]-q[:,None],0)
        excess=np.maximum(stock[:,None]+q[:,None]-d,0)
        costs=(proof["cu"]*shortage+proof["co"]*excess).sum(axis=0)
        np.testing.assert_allclose(costs,proof["losses"],atol=1e-6)
        risk=tail(costs+proof["preference_cost"],proof["probabilities"],proof["alpha"])
        for metric in ["expected","var","cvar","worst"]:
            np.testing.assert_allclose(risk[metric],proof["scores"][metric],atol=1e-6)
        # An independent fixed-loss LP crosschecks fractional-tail integration.
        np.testing.assert_allclose(risk["cvar"],epigraph_cvar(costs+proof["preference_cost"],proof["probabilities"],proof["alpha"])["cvar"],atol=1e-5)
    checks["exact_losses_and_independent_cvar_epigraph"]=True
    static=json.loads((root/"static_solves.json").read_text())
    for row in static:
        if row["method_id"]=="ceiling" and row["status"] not in {"OPTIMAL","INFEASIBLE","FEASIBLE_NOT_PROVEN_OPTIMAL"}:
            raise AssertionError("Unavailable risk-ceiling solver not classified")
    if not any(s["method_id"]=="ceiling" and s["status"]=="INFEASIBLE" for s in static):
        raise AssertionError("Missing explicit infeasible epsilon diagnostic")
    checks["infeasible_risk_ceiling_not_relaxed"]=True
    pd.DataFrame(fair_rows).to_parquet(root/"tables/fairness_slacks.parquet",index=False)
    audit=dict(status="passed",checks=checks,independent_gates=len(checks),daily_rows=count,
               decision_proofs=len(proofs),retiming_events=retimed,independent_cvar_lps=len(proofs))
    (root/"verification.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    return audit
