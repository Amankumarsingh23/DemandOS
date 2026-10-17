-- Question: how complete is price information, and is any positive-sale revenue unknown?
-- COUNT(price) counts known prices; COUNT(*) counts every observation.
select count(*) as rows, count(sell_price) as priced_rows,
    sum(case when sell_price is null then 1 else 0 end) as missing_prices,
    sum(case when units_sold > 0 and sell_price is null then 1 else 0 end) as unknown_revenue_rows,
    sum(observed_revenue) as known_revenue
from main.fact_daily_sales;
