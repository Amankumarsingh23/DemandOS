select count(*) as invalid_count
from {{ source('season2', 'daily_sales') }}
where units_sold is null or not isfinite(units_sold)
   or units_sold < 0 or units_sold <> trunc(units_sold)
   or (sell_price is not null and (
       not isfinite(sell_price) or sell_price <= 0
       or abs(sell_price - cast(cast(sell_price as decimal(12, 4)) as double)) > 0.00000001))
having count(*) > 0
