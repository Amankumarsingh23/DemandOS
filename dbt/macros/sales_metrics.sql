{% macro sales_metrics(item_grain=false) %}
    count(*) as observation_count,
    sum(units_sold) as units_sold,
    sum(observed_revenue) as known_revenue,
    case when count(*) filter (where observed_revenue is null) = 0
         then sum(observed_revenue) else null end as revenue,
    {% if item_grain %}
    max(case when units_sold > 0 then 1 else 0 end) as active_selling_items,
    {% else %}
    count(distinct item_id) filter (where units_sold > 0) as active_selling_items,
    {% endif %}
    count(*) filter (where units_sold = 0) as zero_sales_count,
    count(*) filter (where price_is_known) as priced_observation_count,
    count(*) filter (where not price_is_known) as missing_price_count,
    count(*) filter (where observed_revenue is null) as unknown_revenue_count,
    cast(count(*) filter (where price_is_known) as double) / count(*) as price_completeness
{% endmacro %}
