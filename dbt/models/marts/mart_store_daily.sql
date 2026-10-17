-- Grain/key: store_id, date. Active items means items with positive recorded sales.
select store_id, date, {{ sales_metrics() }}
from {{ ref('fact_daily_sales') }}
group by store_id, date
