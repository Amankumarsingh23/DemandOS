"""Independent business-unit checks on persisted orders and resource balances."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd


def verify_outputs(directory):
    directory=Path(directory);checks={}
    with duckdb.connect() as con:
        con.execute("set threads=2");con.execute("set memory_limit='1GB'");con.execute("set temp_directory=?",[str(directory/"spill")])
        for name in ["daily","allocations","resources","runs","solver_diagnostics","scenarios","origin_allocations","events"]:
            con.read_parquet(str(directory/"tables"/(name+".parquet"))).create_view(name)
        def gate(name,sql):
            count=con.execute(sql).fetchone()[0]
            if count:raise AssertionError(f"Independent gate {name}: {count} failures")
            checks[name]=True
        keys="origin_date,scenario_config_id,method,series_id"
        gate("unique_daily",f"select count(*) from (select {keys},day_index from daily group by all having count(*)<>1)")
        gate("complete_28_days",f"select count(*) from (select {keys} from daily group by all having count(*)<>28 or min(day_index)<>1 or max(day_index)<>28)")
        gate("calendar", "select count(*) from daily where date_diff('day',origin_date,simulation_date)<>day_index")
        gate("stock_demand_backlog", """select count(*) from daily where opening_stock+arrivals<>immediate_served+old_backlog_fulfilled+ending_stock or
            demand_proxy<>immediate_served+lost_units+new_backlog or opening_backlog+new_backlog<>ending_backlog+old_backlog_fulfilled or
            ending_on_order<>on_order_before_action+ordered or least(ending_stock,ending_backlog,ending_on_order,ordered)<0""")
        gate("cost_components","select count(*) from daily where abs(operational_cost-holding_cost-lost_cost-backlog_cost-ordering_cost-procurement_cost+terminal_credit)>1e-7")
        gate("approved_request_pack_moq","select count(*) from allocations where approved<0 or approved>candidate_quantity or approved<>round(approved) or approved%pack<>0 or (approved>0 and approved<moq) or activation<>(approved>0)::integer")
        gate("approved_equals_placed",f"select count(*) from daily d join allocations a using({keys},day_index) where d.ordered<>a.approved or d.position_before_order<>a.position")
        gate("shared_paths_and_shocks","""select count(*) from (select origin_date,scenario_config_id,series_id,day_index from daily group by all
            having count(distinct demand_proxy)<>1 or count(distinct supplier_shock_id)<>1)""")
        gate("persistent_budget","select count(*) from resources where abs(budget_before-committed-budget_after)>1e-7 or (constrained and (budget_after < -1e-7 or flow_used>flow_limit+1e-7))")
        gate("no_budget_reset","""select count(*) from (select *,lag(budget_after) over(partition by origin_date,scenario_config_id,method order by day_index) as prior from resources)
            where (day_index=1 and budget_before<>budget_initial) or (day_index>1 and abs(prior-budget_before)>1e-7)""")
        gate("procurement_matches_commitments","""select count(*) from (select origin_date,scenario_config_id,method,day_index,sum(procurement_cost+ordering_cost) as cash from daily group by all)
            d join resources r using(origin_date,scenario_config_id,method,day_index) where abs(d.cash-r.committed)>1e-7""")
        gate("run_cost_and_fill",f"""select count(*) from (select {keys},sum(operational_cost) as cost,sum(demand_proxy) as demand,sum(immediate_served) as served from daily group by all)
            d join runs r using({keys}) where abs(d.cost-r.operational_cost)>1e-7 or d.demand<>r.demand_units or d.served<>r.immediate_served_units""")
        gate("scenario_probabilities","select count(*) from (select store_id,item_id,origin_date,source,lead_setting from scenarios group by all having abs(sum(probability)-1)>1e-12 or min(probability)<0)")
        gate("origin_nonanticipative_vectors","""select count(*) from (select origin_date,series_id from origin_allocations group by all
            having count(*)<>4 or count(distinct approved)<>1 or min(approved)<0 or max(approved)>greatest(0,min(frozen_target)-13))""")
        gate("forecast_availability_cutoffs","select count(*) from allocations where model_fit_cutoff>origin_date or forecast_availability_cutoff>origin_date")
        gate("order_arrival_lifecycles",f"""select count(*) from (select {keys},order_id from events group by all
            having count(*) filter(where event in ('placed','initial_outstanding'))<>1 or count(*) filter(where event in ('received','terminal_outstanding'))<>1
            or count(distinct quantity)<>1 or count(distinct arrival_day)<>1 or min(lead_time)<1 or
            max(abs(arrival_day-issue_day-lead_time))<>0 or max(case when event='received' then abs(event_day-arrival_day) else 0 end)<>0)""")
        gate("receipt_flow_matches_daily",f"""select count(*) from daily d full join (
            select {keys},event_day as day_index,sum(quantity) as units from events where event='received' group by all) e
            using({keys},day_index) where coalesce(d.arrivals,0)<>coalesce(e.units,0)""")
        n=con.execute("select count(*) from daily").fetchone()[0]
    resources=pd.read_parquet(directory/"tables/resources.parquet")
    allocations=pd.read_parquet(directory/"tables/allocations.parquet")
    groups=dict(tuple(allocations.groupby(["origin_date","scenario_config_id","method","day_index"])))
    for _,data in resources.groupby(["origin_date","scenario_config_id","method"]):
        prior=None
        for r in data.sort_values("day_index").itertuples():
            candidates=groups[(r.origin_date,r.scenario_config_id,r.method,r.day_index)]
            if abs(float(candidates.purchase@candidates.approved+candidates.fee@candidates.activation)-r.committed)>1e-7:raise AssertionError("Independent cash usage")
            if abs(float(candidates.volume@candidates.approved)-r.flow_used)>1e-7:raise AssertionError("Independent flow coefficients")
            initial=json.loads(r.supplier_initial);remaining=json.loads(r.supplier_remaining);used=json.loads(r.supplier_used)
            if prior is None:prior=initial
            actual_supplier=candidates.groupby("supplier").approved.sum().to_dict()
            actual_store=candidates.groupby("store_id").approved.sum().to_dict()
            for name,value in remaining.items():
                if abs(prior[name]-used.get(name,0)-value)>1e-7 or abs(used.get(name,0)-actual_supplier.get(name,0))>1e-7:raise AssertionError("Supplier depletion/reconciliation")
                if r.constrained and value < -1e-7:raise AssertionError("Supplier hard cap")
            for store,limit in json.loads(r.receiving_limit).items():
                if r.constrained and actual_store.get(store,0)>limit+1e-7:raise AssertionError("Shared store receiving cap")
            prior=remaining
    checks["independent_cash_flow_supplier_receiving"]=True
    result=dict(status="passed",checks=checks,independent_gates=len(checks),daily_rows=n,resource_epochs=len(resources),unconstrained_reference_excluded_from_hard_feasibility=True)
    (directory/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
