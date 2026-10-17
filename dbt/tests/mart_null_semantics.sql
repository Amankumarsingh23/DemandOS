{% for mart in ['mart_store_daily', 'mart_category_daily', 'mart_product_daily', 'mart_store_performance'] %}
select '{{ mart }}' as failing_mart, count(*) as invalid_rows
from {{ ref(mart) }}
where revenue is distinct from (case when unknown_revenue_count = 0 then known_revenue else null end)
   or observation_count <= 0 or units_sold < 0 or active_selling_items < 0
   or zero_sales_count < 0 or zero_sales_count > observation_count
   or unknown_revenue_count < 0 or unknown_revenue_count > missing_price_count
   or priced_observation_count + missing_price_count <> observation_count
   or abs(price_completeness - cast(priced_observation_count as double) / observation_count) > 0.000000001
having count(*) > 0
{% if not loop.last %}union all{% endif %}
{% endfor %}
