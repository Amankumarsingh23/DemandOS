-- Fixed selected window; all observed zeros retained. Never infer launch/stock dates.
select store_id, item_id, count(*) as observed_periods,
    sum(units_sold) as units_sold, sum(observed_revenue) as known_revenue,
    count(*) filter (where sell_price is null) as missing_price_count,
    count(*) filter (where observed_revenue is null) as unknown_revenue_count,
    count(*) filter (where units_sold > 0) as positive_periods,
    count(*) filter (where units_sold = 0) as zero_periods,
    avg(units_sold) as mean_daily_units, stddev_pop(units_sold) as std_daily_units,
    var_pop(units_sold) as variance_daily_units,
    quantile_cont(units_sold, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]) as quantiles,
    stddev_pop(units_sold) / nullif(avg(units_sold), 0) as daily_cv,
    count(*) filter (where units_sold = 0) * 1.0 / count(*) as zero_fraction,
    count(*) * 1.0 / nullif(count(*) filter (where units_sold > 0), 0) as adi,
    case when count(*) filter (where units_sold > 0) >= 2 then
        pow(stddev_pop(units_sold) filter (where units_sold > 0)
            / nullif(avg(units_sold) filter (where units_sold > 0), 0), 2)
        else null end as positive_cv2,
    min(date) as first_observed_date, max(date) as last_observed_date,
    min(date) filter (where units_sold > 0) as first_positive_date,
    date_diff('day', min(date), min(date) filter (where units_sold > 0)) as leading_zero_days,
    date_diff('day', min(date) filter (where units_sold > 0), max(date)) + 1 as periods_from_first_positive
from eda_fact where store_id = ?
group by store_id, item_id
order by store_id, item_id
