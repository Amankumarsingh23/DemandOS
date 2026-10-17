-- XYZ uses complete Saturday-Friday Walmart weeks only, at product-across-stores grain.
with complete_weeks as (
    select wm_yr_wk, min(date) as week_start
    from eda_calendar group by wm_yr_wk having count(*) = 7
), weekly as (
    select p.item_id, c.wm_yr_wk, w.week_start, sum(p.units_sold) as units_sold
    from eda_product_daily p
    join eda_calendar c using (date)
    join complete_weeks w using (wm_yr_wk)
    group by p.item_id, c.wm_yr_wk, w.week_start
), stats as (
    select item_id, count(*) as weekly_periods,
        count(*) filter (where units_sold > 0) as positive_weeks,
        avg(units_sold) as mean_weekly_units,
        stddev_pop(units_sold) as std_weekly_units,
        stddev_pop(units_sold) / nullif(avg(units_sold), 0) as weekly_cv,
        min(week_start) filter (where units_sold > 0) as first_positive_week
    from weekly group by item_id
)
select p.item_id, coalesce(s.weekly_periods, 0) as weekly_periods,
    coalesce(s.positive_weeks, 0) as positive_weeks,
    s.mean_weekly_units, s.std_weekly_units, s.weekly_cv, s.first_positive_week
from (select distinct item_id from eda_product_daily) p
left join stats s using (item_id)
order by p.item_id
