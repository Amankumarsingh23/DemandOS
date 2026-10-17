-- Question: which stores sold more than 10,000 units in observed 2016 history?
-- WHERE selects rows before GROUP BY; HAVING selects groups after aggregation.
select store_id, sum(units_sold) as units_sold
from main.fact_daily_sales
where date >= date '2016-01-01'
group by store_id
having sum(units_sold) > 10000
order by units_sold desc, store_id;
