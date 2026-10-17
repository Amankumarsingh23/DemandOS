-- Grain/key: item_id. Conflicting metadata produces duplicate keys and fails tests.
select distinct item_id, dept_id, cat_id
from {{ ref('stg_daily_sales') }}
