{% for mart in ['mart_store_daily', 'mart_category_daily', 'mart_product_daily', 'mart_store_performance'] %}
select '{{ mart }}' as failing_mart
from (
    select sum(observation_count) as rows, sum(units_sold) as units,
        sum(known_revenue) as known_revenue,
        sum(zero_sales_count) as zeros, sum(missing_price_count) as missing,
        sum(unknown_revenue_count) as unknowns
    from {{ ref(mart) }}
) m cross join (
    select count(*) as rows, sum(units_sold) as units, sum(observed_revenue) as known_revenue,
        count(*) filter (where units_sold = 0) as zeros,
        count(*) filter (where sell_price is null) as missing,
        count(*) filter (where observed_revenue is null) as unknowns
    from {{ ref('fact_daily_sales') }}
) f
where m.rows is distinct from f.rows or m.units is distinct from f.units
   or m.known_revenue is distinct from f.known_revenue or m.zeros is distinct from f.zeros
   or m.missing is distinct from f.missing or m.unknowns is distinct from f.unknowns
{% if not loop.last %}union all{% endif %}
{% endfor %}
