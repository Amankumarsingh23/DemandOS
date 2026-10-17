-- No dimension joins or duplicated fact storage. Relationships are tested.
select
    store_id, item_id, dept_id, cat_id, state_id, date, d, wm_yr_wk,
    units_sold, sell_price,
    sell_price is not null as price_is_known,
    -- Zero recorded units imply zero recorded revenue even with unknown price.
    -- Positive sales with unknown price imply unknown recorded revenue.
    case when units_sold = 0 then cast(0 as decimal(30, 4))
         when sell_price is not null then units_sold * sell_price
         else null end as observed_revenue
from {{ ref('stg_daily_sales') }}
