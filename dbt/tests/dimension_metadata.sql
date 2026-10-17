-- Assert metadata consistency using compact distinct projections, not daily joins.
select 'product' as failing_dimension
from (select distinct item_id, dept_id, cat_id from {{ ref('fact_daily_sales') }}) f
left join {{ ref('dim_product') }} p using (item_id)
where f.dept_id is distinct from p.dept_id or f.cat_id is distinct from p.cat_id
union all
select 'store' as failing_dimension
from (select distinct store_id, state_id from {{ ref('fact_daily_sales') }}) f
left join {{ ref('dim_store') }} s using (store_id)
where f.state_id is distinct from s.state_id
union all
select 'calendar' as failing_dimension
from (select distinct date, d, wm_yr_wk from {{ ref('fact_daily_sales') }}) f
left join {{ ref('dim_calendar') }} c using (date)
where f.d is distinct from c.d or f.wm_yr_wk is distinct from c.wm_yr_wk
