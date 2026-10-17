-- Grain: observed calendar date; unique keys date and d.
-- Includes sales-history dates only, not unseen future calendar dates.
select distinct date, d, wm_yr_wk, weekday, wday, month, year,
    event_name_1, event_type_1, event_name_2, event_type_2,
    snap_CA, snap_TX, snap_WI
from {{ ref('stg_daily_sales') }}
