-- Question: what recorded sales occurred at CA_1 at the start of 2016?
select date, store_id, item_id, units_sold, sell_price
from main.fact_daily_sales
where store_id = 'CA_1' and date between date '2016-01-01' and date '2016-01-07'
order by date, item_id
limit 10;
