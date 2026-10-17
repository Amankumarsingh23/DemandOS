-- Grain/key: store_id.
select distinct store_id, state_id
from {{ ref('stg_daily_sales') }}
