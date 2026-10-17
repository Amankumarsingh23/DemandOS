-- Question: how do daily store sales compare with prior observed days?
-- Prior seven ROWS assumes unique continuous daily observations (tested).
-- Historical analysis only: multi-horizon forecasting must respect forecast origin.
with history as (
    select store_id, date, units_sold,
        lag(units_sold) over (partition by store_id order by date) as previous_day_units,
        avg(units_sold) over (
            partition by store_id order by date
            rows between 7 preceding and 1 preceding
        ) as prior_seven_day_average,
        count(*) over (
            partition by store_id order by date
            rows between 7 preceding and 1 preceding
        ) as prior_days_available
    from main.mart_store_daily
)
select * from history
where store_id = 'CA_1' and date between date '2016-01-01' and date '2016-01-14'
order by date;
