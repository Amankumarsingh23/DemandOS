-- Question: which ten products sold the most recorded units across the history?
with product_sales as (
    select item_id, sum(units_sold) as units
    from main.mart_product_daily
    group by item_id
), ranked as (
    select *, dense_rank() over (order by units desc) as sales_rank
    from product_sales
)
select * from ranked
where sales_rank <= 10
order by sales_rank, item_id;
