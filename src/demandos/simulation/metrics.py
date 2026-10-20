"""Operational ledgers and paired series-origin clusters, never daily IID claims."""
import numpy as np
import pandas as pd

from .engine import RUN_KEYS, STATUS

COST_COLUMNS = ["holding_cost", "lost_cost", "backlog_cost", "ordering_cost", "procurement_cost", "terminal_credit"]


def run_metrics(daily):
    scored = daily.loc[daily.scored]
    if scored.empty: raise ValueError("No scored simulation days")
    demand = int(scored.demand_proxy.sum())
    served = int(scored.immediate_served.sum())
    delayed = int(scored.old_backlog_fulfilled.sum())
    total = float(scored.operational_cost.sum())
    return dict(scored_days=len(scored), warmup_days=int((~daily.scored).sum()), demand_units=demand,
        immediate_served_units=served, immediate_fill_rate=served/demand if demand else None,
        daily_no_shortage_fraction=float(scored.no_shortage.mean()), lost_units=int(scored.lost_units.sum()),
        backorders_created=int(scored.new_backlog.sum()), backorders_fulfilled=delayed,
        ending_backlog=int(daily.ending_backlog.iloc[-1]), backlog_unit_days=int(scored.ending_backlog.sum()),
        fulfilled_wait_unit_days=int(scored.fulfilled_backorder_wait_unit_days.sum()),
        fulfilled_backorder_mean_wait=float(scored.fulfilled_backorder_wait_unit_days.sum()/delayed) if delayed else None,
        ending_stock=int(daily.ending_stock.iloc[-1]), ending_on_order=int(daily.ending_on_order.iloc[-1]),
        stock_unit_days=int(scored.ending_stock.sum()), orders_placed=int(scored.ordered.gt(0).sum()),
        ordered_units=int(scored.ordered.sum()), received_units=int(scored.arrivals.sum()), operational_cost=total,
        cost_per_demanded_unit=total/demand if demand else None,
        cost_per_immediately_served_unit=total/served if served else None,
        replenishment_csl=None, csl_status="not_defined_no_cycle_segmentation",
        **{name: float(scored[name].sum()) for name in COST_COLUMNS})


def aggregate(runs, columns):
    rows = []
    for key, group in runs.groupby(columns, dropna=False):
        demand, served = group.demand_units.sum(), group.immediate_served_units.sum()
        cost = group.operational_cost.sum()
        rows.append(dict(zip(columns, key if isinstance(key, tuple) else (key,)), runs=len(group),
            demand_units=int(demand), immediate_served_units=int(served), cost_mean=float(group.operational_cost.mean()),
            cost_total=float(cost), cost_per_demanded_unit=float(cost/demand) if demand else None,
            demand_weighted_fill=float(served/demand) if demand else None,
            unweighted_fill=float(group.immediate_fill_rate.mean()) if group.immediate_fill_rate.notna().any() else None,
            no_demand_runs=int(group.demand_units.eq(0).sum()),
            daily_no_shortage_fraction=float(group.daily_no_shortage_fraction.mean()),
            lost_mean=float(group.lost_units.mean()), ending_stock_mean=float(group.ending_stock.mean()),
            ending_on_order_mean=float(group.ending_on_order.mean()), ending_backlog_mean=float(group.ending_backlog.mean()),
            backlog_unit_days_mean=float(group.backlog_unit_days.mean()), orders_mean=float(group.orders_placed.mean()),
            ordered_units_mean=float(group.ordered_units.mean()), stock_unit_days_mean=float(group.stock_unit_days.mean()),
            **{name: float(group[name].mean()) for name in COST_COLUMNS}, status=STATUS))
    return pd.DataFrame(rows)


PAIR_KEYS = ["experiment_id", "series_id", "origin_date", "cost_scenario_id", "lead_scenario_id", "demand_scenario_id", "replication_id"]


def paired(runs):
    """Reference historical-fractile RS, with exact identical exogenous cases."""
    if runs.duplicated(RUN_KEYS).any(): raise ValueError("Duplicate canonical run keys")
    outputs = []
    for experiment, data in runs.groupby("experiment_id"):
        reference = data[data.source.eq("historical_fractile") & data.rule.eq("RS")]
        if reference.empty or reference.duplicated(PAIR_KEYS).any(): raise ValueError("Missing/duplicated paired reference")
        wanted = set(reference[PAIR_KEYS].itertuples(index=False, name=None))
        for policy, group in data.groupby("policy_id"):
            if set(group[PAIR_KEYS].itertuples(index=False, name=None)) != wanted:
                raise ValueError("Unmatched policy cases")
            comparison = group.merge(reference[PAIR_KEYS+["operational_cost", "immediate_fill_rate"]], on=PAIR_KEYS,
                                     suffixes=("", "_reference"), validate="one_to_one")
            comparison["cost_difference"] = comparison.operational_cost-comparison.operational_cost_reference
            comparison["fill_difference"] = comparison.immediate_fill_rate-comparison.immediate_fill_rate_reference
            comparison["reference_policy"] = "historical_fractile:RS"
            outputs.append(comparison)
    pairs = pd.concat(outputs, ignore_index=True)
    summaries = []
    columns = ["experiment_id", "cohort", "cost_scenario_id", "lead_scenario_id", "source", "rule"]
    for key, group in pairs.groupby(columns):
        # Supplier replications are averaged WITHIN series-origin clusters first.
        clustered = group.groupby(["series_id", "origin_date"]).cost_difference.mean()
        n = len(clustered)
        summaries.append(dict(zip(columns, key), pairs=len(group), clusters=n,
            mean_cost_difference=float(clustered.mean()),
            descriptive_cluster_se=float(clustered.std(ddof=1)/np.sqrt(n)) if n >= 3 else None,
            status=STATUS))
    return pairs, pd.DataFrame(summaries)
