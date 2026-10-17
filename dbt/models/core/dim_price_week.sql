-- Grain/key: store_id, item_id, wm_yr_wk.
-- Only known prices represented in the observed sales history, not all raw weeks.
select distinct store_id, item_id, wm_yr_wk, sell_price
from {{ ref('stg_daily_sales') }}
where sell_price is not null
