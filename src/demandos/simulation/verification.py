"""Independent bounded SQL reconciliation of persisted sequential artifacts."""
import json
from pathlib import Path
import duckdb

from .engine import RUN_KEYS


def verify(directory):
    directory = Path(directory)
    keys = ",".join(RUN_KEYS)
    checks = {}
    with duckdb.connect() as con:
        con.execute("set threads=2"); con.execute("set memory_limit='1GB'")
        con.execute("set temp_directory=?", [str(directory/"spill")])
        for name in ["daily_ledger", "order_events", "exogenous_demand", "supplier_shocks", "paired_daily_costs", "paired_runs", "runs"]:
            con.read_parquet(str(directory/"tables"/(name+".parquet"))).create_view(name)
        def gate(name, sql):
            failures = con.execute(sql).fetchone()[0]
            if failures: raise AssertionError(f"Saved artifact hard gate failed: {name}: {failures}")
            checks[name] = True
        gate("unique_daily_keys", f"select count(*) from (select {keys},day_index from daily_ledger group by all having count(*)<>1)")
        gate("unique_run_keys", f"select count(*) from (select {keys} from runs group by all having count(*)<>1)")
        gate("daily_conservation", """select count(*) from daily_ledger where
            opening_stock+arrivals<>immediate_served+old_backlog_fulfilled+ending_stock or
            demand_proxy<>immediate_served+new_backlog+lost_units or
            opening_backlog+new_backlog<>old_backlog_fulfilled+ending_backlog or
            ending_on_order<>on_order_before_action+ordered or
            position_before_order<>ending_stock+on_order_before_action-ending_backlog or
            position_after_order<>ending_stock+ending_on_order-ending_backlog or
            least(ending_stock,ending_on_order,ending_backlog,lost_units,ordered)<0 or
            (mode='lost_sales' and (ending_backlog<>0 or new_backlog<>0)) or
            (mode='backorders' and lost_units<>0)""")
        gate("cost_components", """select count(*) from daily_ledger where abs(operational_cost-
            (holding_cost+lost_cost+backlog_cost+ordering_cost+procurement_cost-terminal_credit))>1e-8""")
        gate("calendar_and_leap_alignment", "select count(*) from daily_ledger where date_diff('day',origin_date,simulation_date)<>calendar_day")
        gate("complete_daily_paths", f"select count(*) from (select {keys} from daily_ledger group by all having min(day_index)<>1 or max(day_index)<>count(*) or max(calendar_day)-min(calendar_day)+1<>count(*))")
        gate("state_carryover", f"""select count(*) from (select *,
            lag(ending_stock) over w as prior_stock, lag(ending_backlog) over w as prior_backlog,
            lag(ending_on_order) over w as prior_pipeline from daily_ledger
            window w as (partition by {keys} order by day_index))
            where (day_index=1 and (opening_stock<>initial_on_hand or opening_on_order<>initial_on_order)) or
            (day_index>1 and (opening_stock<>prior_stock or opening_backlog<>prior_backlog or opening_on_order<>prior_pipeline))""")
        gate("event_uniqueness", f"select count(*) from (select {keys},order_id,event from order_events group by all having count(*)<>1)")
        gate("delivery_timing", """select count(*) from order_events where arrival_day<>issue_day+lead_time or lead_time<1 or quantity<=0 or
            (event='received' and event_day<>arrival_day) or (event='placed' and event_day<>issue_day) or
            (event='terminal_outstanding' and arrival_day<=event_day)""")
        gate("individual_order_lifecycle", f"""select count(*) from (select {keys},order_id from order_events group by all
            having count(*) filter(where event in ('placed','initial_outstanding'))<>1 or
            count(*) filter(where event in ('received','terminal_outstanding'))<>1 or count(distinct quantity)<>1 or count(distinct arrival_day)<>1)""")
        gate("queue_units_reconcile", f"""select count(*) from (select {keys},
            sum(case when event in ('placed','initial_outstanding') then quantity else -quantity end) as gap
            from order_events group by all having gap<>0)""")
        gate("arrivals_match_daily", f"""select count(*) from (
            select {keys},day_index,arrivals from daily_ledger) d full join (
            select {keys},event_day as day_index,sum(quantity) as receipts from order_events where event='received' group by all) e
            using({keys},day_index) where coalesce(arrivals,0)<>coalesce(receipts,0)""")
        gate("orders_match_daily", f"""select count(*) from (
            select {keys},day_index,ordered from daily_ledger) d full join (
            select {keys},event_day as day_index,sum(quantity) as units from order_events where event='placed' group by all) e
            using({keys},day_index) where coalesce(ordered,0)<>coalesce(units,0)""")
        gate("unique_demand_scenarios", "select count(*) from (select series_id,origin_date,replication_id,demand_scenario_id,simulation_date from exogenous_demand group by all having count(*)<>1)")
        gate("unique_supplier_scenarios", "select count(*) from (select series_id,origin_date,replication_id,lead_scenario_id,calendar_day from supplier_shocks group by all having count(*)<>1)")
        gate("common_exogenous_comparisons", """select count(*) from (select experiment_id,series_id,origin_date,
            cost_scenario_id,lead_scenario_id,demand_scenario_id,replication_id,day_index from daily_ledger group by all
            having count(distinct demand_proxy)<>1 or count(distinct supplier_shock_id)<>1 or count(distinct sampled_lead)<>1 or
            count(distinct initial_on_hand)<>1 or count(distinct initial_on_order)<>1 or count(distinct mode)<>1)""")
        gate("run_cost_and_service_reconciliation", f"""select count(*) from (
            select {keys},sum(operational_cost) as cost, sum(demand_proxy) as demand,
            sum(immediate_served) as immediate, sum(lost_units) as lost,
            sum(new_backlog) as created, sum(old_backlog_fulfilled) as delayed,
            sum(ending_stock) as exposure, sum(ending_backlog) as backlog_days,
            sum(ordered) as orders, count(*) filter(where ordered>0) as order_count,
            sum(arrivals) as received,avg(no_shortage::integer) as daily_service,
            arg_max(ending_stock,day_index) as final_stock,arg_max(ending_on_order,day_index) as final_pipeline,
            arg_max(ending_backlog,day_index) as final_backlog
            from daily_ledger where scored group by all) d join runs r using({keys})
            where abs(d.cost-r.operational_cost)>1e-7 or d.demand<>r.demand_units or d.immediate<>r.immediate_served_units
            or d.lost<>r.lost_units or d.created<>r.backorders_created or d.delayed<>r.backorders_fulfilled
            or d.exposure<>r.stock_unit_days or d.backlog_days<>r.backlog_unit_days or d.orders<>r.ordered_units
            or d.order_count<>r.orders_placed or d.received<>r.received_units or abs(d.daily_service-r.daily_no_shortage_fraction)>1e-12
            or d.final_stock<>r.ending_stock or d.final_pipeline<>r.ending_on_order or d.final_backlog<>r.ending_backlog
            or (d.demand=0 and r.immediate_fill_rate is not null)
            or (d.demand>0 and abs(d.immediate::double/d.demand-r.immediate_fill_rate)>1e-12)""")
        gate("daily_demand_matches_exogenous", """select count(*) from daily_ledger d left join exogenous_demand e
            on d.series_id=e.series_id and d.origin_date=e.origin_date and d.demand_scenario_id=e.demand_scenario_id
            and d.simulation_date=e.simulation_date and
            (case when d.demand_scenario_id='m5_recorded_sales_proxy' then 0 else d.replication_id end)=e.replication_id
            where e.demand_proxy is null or d.demand_proxy<>e.demand_proxy""")
        gate("paired_daily_to_run_cost", f"""select count(*) from (select {keys},sum(cost_difference) as daily_difference from paired_daily_costs where scored group by all) d
            join paired_runs r using({keys})
            where abs(daily_difference-(operational_cost-operational_cost_reference))>1e-7""")
        totals = con.execute("select count(*),count(*) filter(where experiment_id='main') from daily_ledger").fetchone()
    result = dict(status="passed", independent_sql_gates=len(checks), checks=checks, daily_rows=totals[0], main_daily_rows=totals[1])
    (directory/"verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result
