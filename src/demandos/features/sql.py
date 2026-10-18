"""Compact SQL reductions. No horizon cross join touches the full fact table."""

import numpy as np
import pandas as pd

from demandos.eda.queries import fetch_frame
from .contracts import origin_dictionary, cast_frame


def origin_sql(config):
    lag_columns = []
    for lag in config.lags:
        lag_columns += [f"max(s.units_sold) filter(where ago={lag}) as origin_lag_{lag}",
                        f"max(s.date) filter(where ago={lag}) as origin_lag_{lag}_source_date"]
    windows = []
    for window in config.windows:
        condition = f"ago between 0 and {window - 1}"
        complete = f"count(s.date) filter(where {condition})={window}"
        expressions = {"mean": "avg(s.units_sold)", "sum": "sum(s.units_sold)", "std": "stddev_pop(s.units_sold)",
                       "max": "max(s.units_sold)", "zero_fraction": "avg(cast(s.units_sold=0 as double))",
                       "positive_count": "sum(cast(s.units_sold>0 as bigint))"}
        windows += [f"case when {complete} then {expression} filter(where {condition}) end as rolling_{stat}_{window}_at_origin"
                    for stat, expression in expressions.items()]
        windows.append(f"count(s.date) filter(where {condition}) as rolling_observations_{window}")
    return f"""with p as (select cast(? as date) as origin_date),
    s as (select f.store_id,f.item_id,f.date,f.units_sold,date_diff('day',f.date,p.origin_date) as ago
        from feature_sales f join feature_keys k using(store_id,item_id) cross join p
        where f.date between p.origin_date - {config.lookback} * interval '1 day' and p.origin_date),
    a as (select k.store_id,k.state_id,k.item_id,k.dept_id,k.cat_id,p.origin_date,k.first_observed_date,
        min(s.date) as bounded_history_start,max(s.date) as max_sales_source_date,
        count(s.date) as history_observations,count(distinct s.date) as history_distinct_dates,
        count(*) filter(where s.date is not null and (s.units_sold is null or not isfinite(cast(s.units_sold as double))
            or s.units_sold<0 or s.units_sold<>floor(s.units_sold))) as history_invalid_rows,
        max(s.units_sold) filter(where ago=0) as sales_at_origin,
        {', '.join(lag_columns + windows)},
        case when count(s.date) filter(where ago between 7 and 13)=7 then avg(s.units_sold) filter(where ago between 7 and 13) end as mean_previous_7_at_origin
        from feature_keys k cross join p left join s using(store_id,item_id)
        group by all)
    select *, rolling_mean_7_at_origin-mean_previous_7_at_origin as level_change_7,
        (rolling_mean_7_at_origin-mean_previous_7_at_origin)/nullif(mean_previous_7_at_origin,0) as relative_level_change_7
    from a order by store_id,item_id"""


PRICE_COLUMNS = ["store_id", "item_id", "wm_yr_wk", "observed_at", "available_at"]


def validate_price_evidence(con, evidence):
    """Join explicit historical knowledge stamps to the canonical weekly price key."""
    if evidence is None:
        return None
    if set(evidence) != set(PRICE_COLUMNS) or evidence.empty:
        raise ValueError("Historical price evidence requires exact nonempty key/observation/availability columns")
    frame = evidence.copy()
    if frame.isna().any().any():
        raise ValueError("Incomplete historical-price provenance")
    for name in ("observed_at", "available_at"):
        frame[name] = pd.to_datetime(frame[name])
        if not frame[name].eq(frame[name].dt.normalize()).all():
            raise ValueError("Historical-price knowledge stamps must be daily midnight dates")
    if frame.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("One evidence record per store-item-week required")
    con.register("feature_price_evidence", frame)
    joined = fetch_frame(con, """select e.*, p.sell_price, c.wm_yr_wk as observed_week
        from feature_price_evidence e left join feature_prices p using(store_id,item_id,wm_yr_wk)
        left join feature_calendar c on c.date=e.observed_at""")
    if len(joined) != len(frame) or joined.sell_price.isna().any() or joined.observed_week.isna().any():
        raise ValueError("Historical price/calendar join missing or multiplying records")
    if not joined.wm_yr_wk.eq(joined.observed_week).all():
        raise ValueError("Price observation date does not match its historical week")
    if not np.isfinite(joined.sell_price.to_numpy(float)).all() or joined.sell_price.lt(0).any():
        raise ValueError("Invalid historical selling prices")
    return joined.drop(columns="observed_week")


def price_features(keys, origin, verified=None):
    """No retrospective price enters without BOTH origin-safe knowledge stamps."""
    origin = pd.Timestamp(origin)
    records = []
    selected = None if verified is None else verified[
        pd.to_datetime(verified.observed_at).le(origin) & pd.to_datetime(verified.available_at).le(origin)]
    groups = ({} if selected is None else
              {key: group.sort_values(["observed_at", "wm_yr_wk"]) for key, group in selected.groupby(["store_id", "item_id"], sort=False)})
    for key in keys.itertuples():
        record = {"store_id": key.store_id, "item_id": key.item_id,
                  **{column: np.nan for column in ("last_known_price", "prior_known_price", "historical_price_change", "historical_relative_price_4")},
                  **{column: pd.NaT for column in ("last_price_observed_at", "last_price_available_at", "prior_price_observed_at", "prior_price_available_at")},
                  "known_price_weeks": 0, "last_price_missing": True, "prior_price_missing": True}
        rows = groups.get((key.store_id, key.item_id))
        if rows is not None:
            record["known_price_weeks"] = len(rows)
            for offset, prefix in ((1, "last"), (2, "prior")):
                if len(rows) >= offset:
                    row = rows.iloc[-offset]
                    record[f"{prefix}_known_price"] = float(row.sell_price)
                    record[f"{prefix}_price_observed_at"] = pd.Timestamp(row.observed_at)
                    record[f"{prefix}_price_available_at"] = pd.Timestamp(row.available_at)
                    record[f"{prefix}_price_missing"] = False
            previous = record["prior_known_price"]
            if pd.notna(previous) and previous > 0:
                record["historical_price_change"] = record["last_known_price"] / previous - 1
            if len(rows) >= 4 and float(rows.tail(4).sell_price.mean()) > 0:
                record["historical_relative_price_4"] = record["last_known_price"] / float(rows.tail(4).sell_price.mean())
        records.append(record)
    return pd.DataFrame(records)


def origin_features(con, origin, config, verified_prices=None):
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or origin != origin.normalize():
        raise ValueError("A midnight end-of-day origin date is required")
    frame = fetch_frame(con, origin_sql(config), [origin.date()])
    if frame.empty or frame.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Missing or duplicate origin-level keys")
    expected = ((origin - pd.to_datetime(frame.first_observed_date)).dt.days + 1).clip(upper=config.lookback + 1)
    if frame.sales_at_origin.isna().any() or not frame.history_observations.eq(expected).all():
        raise ValueError("Incomplete daily history or missing origin observation")
    if not frame.history_observations.eq(frame.history_distinct_dates).all() or frame.history_invalid_rows.ne(0).any():
        raise ValueError("Duplicate, nonfinite or invalid daily sales observations")
    if not pd.to_datetime(frame.max_sales_source_date).eq(origin).all():
        raise ValueError("Historical source extends beyond or falls short of origin")
    # With no evidence, this remains a compact SQL-only full-panel reduction.
    if verified_prices is None:
        for name in ("last_known_price", "prior_known_price", "historical_price_change", "historical_relative_price_4"):
            frame[name] = np.nan
        for name in ("last_price_observed_at", "last_price_available_at", "prior_price_observed_at", "prior_price_available_at"):
            frame[name] = pd.NaT
        frame["known_price_weeks"], frame["last_price_missing"], frame["prior_price_missing"] = 0, True, True
    else:
        prices = price_features(frame[["store_id", "item_id"]], origin, verified_prices)
        frame = frame.merge(prices, on=["store_id", "item_id"], validate="one_to_one")
    return cast_frame(frame, origin_dictionary(config))


def expand_calendar(con, origins, horizons=tuple(range(1, 29))):
    if not horizons or len(set(horizons)) != len(horizons) or any(type(h) is not int or not 1 <= h <= 28 for h in horizons):
        raise ValueError("Unique integer horizons 1..28 required")
    if origins.duplicated(["store_id", "item_id", "origin_date"]).any():
        raise ValueError("Duplicate origin rows")
    if not set(origins.state_id).issubset({"CA", "TX", "WI"}):
        raise ValueError("No state-specific SNAP mapping for unknown state")
    con.register("feature_origin_rows", origins)
    con.register("feature_horizons", pd.DataFrame({"horizon": list(horizons)}))
    frame = fetch_frame(con, """select o.*, h.horizon, cast(o.origin_date + h.horizon * interval '1 day' as date) as target_date,
        c.date as calendar_match, cast(isodow(c.date)-1 as bigint) as target_weekday, cast(month(c.date) as bigint) as target_month,
        isodow(c.date) in (6,7) as target_weekend,
        cast(c.event_name_1 is not null as bigint)+cast(c.event_name_2 is not null as bigint) as target_event_count,
        c.event_name_1 is not null or c.event_name_2 is not null as target_has_event,
        coalesce(c.event_type_1='Cultural' or c.event_type_2='Cultural',false) as target_event_cultural,
        coalesce(c.event_type_1='National' or c.event_type_2='National',false) as target_event_national,
        coalesce(c.event_type_1='Religious' or c.event_type_2='Religious',false) as target_event_religious,
        coalesce(c.event_type_1='Sporting' or c.event_type_2='Sporting',false) as target_event_sporting,
        case o.state_id when 'CA' then c.snap_CA when 'TX' then c.snap_TX when 'WI' then c.snap_WI end as target_snap,
        sin(2*pi()*(isodow(c.date)-1)/7) as target_weekday_sin, cos(2*pi()*(isodow(c.date)-1)/7) as target_weekday_cos,
        sin(2*pi()*(month(c.date)-1)/12) as target_month_sin, cos(2*pi()*(month(c.date)-1)/12) as target_month_cos
        from feature_origin_rows o cross join feature_horizons h
        left join feature_calendar c on c.date=cast(o.origin_date+h.horizon*interval '1 day' as date)
        order by o.origin_date,o.store_id,o.item_id,h.horizon""")
    if len(frame) != len(origins) * len(horizons) or frame.calendar_match.isna().any():
        raise ValueError("Target calendar missing or multiplying feature rows")
    if frame.target_snap.isna().any() or not frame.target_snap.isin([0, 1]).all():
        raise ValueError("Invalid state-specific SNAP calendar")
    return frame.drop(columns="calendar_match")
