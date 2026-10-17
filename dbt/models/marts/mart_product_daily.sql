-- Grain/key: item_id, date, aggregated across all stores.
-- At this grain item_id is constant, and each store contributes one tested fact
-- row per day. Avoid large DISTINCT aggregate states for equivalent counts.
select item_id, date, {{ sales_metrics(item_grain=true) }},
    count(*) filter (where units_sold > 0) as selling_store_count
from {{ ref('fact_daily_sales') }}
group by item_id, date
