-- A failure returns one summary row, never the full fact.
with source_totals as (
    select count(*) as rows, sum(units_sold) as units
    from {{ source('season2', 'daily_sales') }}
), fact_totals as (
    select count(*) as rows, sum(units_sold) as units
    from {{ ref('fact_daily_sales') }}
)
select * from source_totals s cross join fact_totals f
where s.rows <> f.rows or s.units <> f.units
   or f.rows <> {{ var('expected_fact_rows') }}
