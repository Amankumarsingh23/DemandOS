-- Grain: one observed store/item/day, inherited from verified Season 2.
select
    id, item_id, dept_id, cat_id, store_id, state_id, d,
    cast(units_sold as bigint) as units_sold,
    cast(date as date) as date,
    wm_yr_wk, weekday, wday, month, year,
    event_name_1, event_type_1, event_name_2, event_type_2,
    snap_CA, snap_TX, snap_WI,
    -- Four decimal places preserve the current M5 prices; tested against source.
    cast(sell_price as decimal(12, 4)) as sell_price
from {{ source('season2', 'daily_sales') }}
