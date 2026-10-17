-- Grain/key: store_id over the entire observed history.
select store_id, {{ sales_metrics() }},
    min(date) as first_sales_date, max(date) as last_sales_date,
    count(distinct date) as observed_days
from {{ ref('fact_daily_sales') }}
group by store_id
