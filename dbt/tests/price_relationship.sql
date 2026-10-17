-- Validate known price references, including exact value, at weekly grain.
select f.store_id, f.item_id, f.wm_yr_wk
from (
    select distinct store_id, item_id, wm_yr_wk, sell_price
    from {{ ref('fact_daily_sales') }} where sell_price is not null
) f
left join {{ ref('dim_price_week') }} p
    using (store_id, item_id, wm_yr_wk)
where p.sell_price is null or p.sell_price <> f.sell_price
