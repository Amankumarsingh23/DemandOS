-- Grain/key: cat_id, date, aggregated across all stores.
select cat_id, date, {{ sales_metrics() }}
from {{ ref('fact_daily_sales') }}
group by cat_id, date
