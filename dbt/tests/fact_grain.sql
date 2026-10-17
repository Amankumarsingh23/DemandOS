-- Grouped distinct dates establish exact daily uniqueness without returning 58M keys.
select store_id, item_id, count(*) as rows, count(distinct date) as dates
from {{ ref('fact_daily_sales') }}
group by store_id, item_id
having count(*) <> count(distinct date)
    or count(*) <> {{ var('expected_sales_days') }}
