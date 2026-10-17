select count(*) as invalid_count
from {{ ref('fact_daily_sales') }}
where store_id is null or item_id is null or date is null or d is null
   or dept_id is null or cat_id is null or state_id is null or wm_yr_wk is null
   or units_sold is null or units_sold < 0
   or (sell_price is not null and (sell_price <= 0 or not isfinite(sell_price)))
   or price_is_known is distinct from (sell_price is not null)
   or observed_revenue is distinct from
        case when units_sold = 0 then 0
             when sell_price is not null then units_sold * sell_price else null end
having count(*) > 0
