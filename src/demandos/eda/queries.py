"""DuckDB reductions only: never fetch full facts or product/day matrices."""

import pandas as pd


SUM_COLUMNS = ("observation_count", "units_sold", "known_revenue", "zero_sales_count",
               "missing_price_count", "unknown_revenue_count")


def fetch_frame(con, sql, parameters=None):
    cursor = con.execute(sql, parameters or [])
    # Decimal revenues remain exact Python Decimals rather than becoming floats.
    return pd.DataFrame.from_records(cursor.fetchall(), columns=[field[0] for field in cursor.description])


def create_views(con, start, end):
    # Dates are validated ISO dates, not user-supplied SQL fragments.
    condition = f"date between date '{start.isoformat()}' and date '{end.isoformat()}'"
    con.execute(f"create or replace temp view eda_fact as select * from main.fact_daily_sales where {condition}")
    con.execute(f"create or replace temp view eda_calendar as select * from main.dim_calendar where {condition}")
    for grain in ("store", "category", "product"):
        con.execute(f"create or replace temp view eda_{grain}_daily as select * from main.mart_{grain}_daily where {condition}")
    sums = ', '.join(f'sum(m.{name}) as {name}' for name in SUM_COLUMNS)
    con.execute(f"""create or replace temp view eda_state_daily as
        select s.state_id, m.date, {sums}
        from eda_store_daily m join main.dim_store s using (store_id)
        group by s.state_id, m.date""")
    con.execute(f"""create or replace temp view eda_overall_daily as
        select m.date, {sums} from eda_store_daily m group by m.date""")


def group_summary(con, grain, key):
    return fetch_frame(con, f"""
        select {key}, count(*) as observed_days,
            sum(observation_count) as observation_count,
            sum(units_sold) as units_sold, sum(known_revenue) as known_revenue,
            sum(zero_sales_count) as zero_sales_count,
            sum(missing_price_count) as missing_price_count,
            sum(unknown_revenue_count) as unknown_revenue_count,
            avg(units_sold) as mean_daily_units, stddev_pop(units_sold) as std_daily_units,
            var_pop(units_sold) as variance_daily_units,
            quantile_cont(units_sold, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]) as quantiles,
            stddev_pop(units_sold) / nullif(avg(units_sold), 0) as daily_cv,
            count(*) filter (where units_sold > 0) as positive_days,
            sum(zero_sales_count) * 1.0 / sum(observation_count) as zero_observation_fraction,
            sum(units_sold) * 1.0 / sum(observation_count) as units_per_observed_series_day
        from eda_{grain}_daily group by {key} order by {key}
    """)


def expand_quantiles(frame):
    result = frame.drop(columns="quantiles").copy()
    result[["p10", "p25", "median_daily_units", "p75", "p90", "p95"]] = pd.DataFrame(frame.quantiles.tolist(), index=frame.index)
    return result
