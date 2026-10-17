-- Question: which category and known weekly price describe these observations?
-- INNER JOIN is safe only after tested product relationships and unique keys.
-- LEFT JOIN preserves rows lacking a weekly price; it does not cure duplicates.
select f.date, f.store_id, f.item_id, p.cat_id, f.units_sold, w.sell_price
from main.fact_daily_sales f
inner join main.dim_product p using (item_id)
left join main.dim_price_week w using (store_id, item_id, wm_yr_wk)
where f.date = date '2016-01-01' and f.store_id = 'CA_1'
order by f.item_id
limit 10;
