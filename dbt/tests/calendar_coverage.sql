with coverage as (
    select count(*) as days, min(date) as first_date, max(date) as last_date,
        count(distinct d) as day_ids,
        count(*) filter (where d <> 'd_' || cast(date_diff('day', min_date, date) + 1 as varchar)) as bad_mapping
    from (
        select *, min(date) over () as min_date
        from {{ ref('dim_calendar') }}
    ) c
)
select * from coverage
where days <> {{ var('expected_sales_days') }} or days <> day_ids
   or date_diff('day', first_date, last_date) + 1 <> days or bad_mapping <> 0
