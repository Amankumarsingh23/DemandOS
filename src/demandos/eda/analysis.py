"""Small deterministic analytics over SQL-reduced frames."""

from decimal import Decimal
import math

import numpy as np
import pandas as pd

from .statistics import abc_classification, combine_segments, intermittent_class, xyz_class


def product_segments(products, weekly, config):
    abc = abc_classification(products, config)
    xyz = weekly.copy()
    xyz[["xyz_class", "xyz_reason"]] = pd.DataFrame([
        xyz_class(row.weekly_periods, row.positive_weeks, row.mean_weekly_units, row.std_weekly_units, config)
        for row in xyz.itertuples()
    ], index=xyz.index)
    return combine_segments(abc, xyz)


def series_segments(series, config):
    result = series.copy()
    result[["regime", "regime_reason"]] = pd.DataFrame([
        intermittent_class(row.observed_periods, row.positive_periods, row.positive_cv2, config)
        for row in result.itertuples()
    ], index=result.index)
    result["regime_after_first_positive"] = [
        intermittent_class(row.periods_from_first_positive, row.positive_periods, row.positive_cv2, config)[0]
        if pd.notna(row.periods_from_first_positive) else "unclassified"
        for row in result.itertuples()
    ]
    result["first_positive_sensitivity_changed"] = result.regime.ne(result.regime_after_first_positive)
    return result


def segment_totals(frame, key, count_name, units_total, revenue_total):
    grouped = frame.groupby(key, dropna=False).agg(**{
        count_name: (key, "size"), "units_sold": ("units_sold", "sum"),
        "known_revenue": ("known_revenue", lambda values: sum((v for v in values if pd.notna(v)), Decimal(0))),
    }).reset_index()
    grouped["volume_share"] = grouped.units_sold / units_total if units_total else np.nan
    grouped["revenue_share"] = grouped.known_revenue.map(lambda value: float(value / revenue_total) if revenue_total else np.nan)
    return grouped


def abc_xyz_matrix(segments, units_total, revenue_total):
    totals = segment_totals(segments, "abc_xyz", "products", units_total, revenue_total)
    index = [a + x for a in "ABC" for x in "XYZ"]
    matrix = totals.set_index("abc_xyz").reindex(index).reset_index()
    for column in ("products", "units_sold", "known_revenue", "volume_share", "revenue_share"):
        matrix[column] = matrix[column].fillna(0)
    matrix["products"] = matrix.products.astype(int)
    return matrix, totals[totals.abc_xyz.eq("unclassified")]


def temporal_tables(daily, store_daily, category_daily):
    daily = daily.sort_values("date").copy()
    daily["date"] = pd.to_datetime(daily.date)
    daily["weekday_number"] = daily.date.dt.dayofweek
    daily["weekday"] = daily.date.dt.day_name()
    daily["month_number"] = daily.date.dt.month
    daily["year"] = daily.date.dt.year
    daily["any_event"] = daily.event_name_1.notna() | daily.event_name_2.notna()
    weekdays = daily.groupby(["weekday_number", "weekday"]).agg(
        mean_daily_units=("units_sold", "mean"), observed_days=("date", "size")
    ).reset_index()
    weekdays["weekday_index"] = weekdays.mean_daily_units / daily.units_sold.mean()
    months = daily.groupby("month_number").agg(mean_daily_units=("units_sold", "mean"), observed_days=("date", "size")).reset_index()
    months["month_index"] = months.mean_daily_units / daily.units_sold.mean()
    stability = daily.groupby(["year", "weekday_number", "weekday"]).agg(mean_daily_units=("units_sold", "mean"), observed_days=("date", "size")).reset_index()
    stability["weekday_index"] = stability.mean_daily_units / stability.year.map(daily.groupby("year").units_sold.mean())
    events = daily.merge(weekdays[["weekday_number", "mean_daily_units"]], on="weekday_number", validate="many_to_one")
    events["weekday_adjusted_index"] = events.units_sold / events.mean_daily_units
    event_summary = events.groupby("any_event").agg(observed_days=("date", "size"), mean_daily_units=("units_sold", "mean"), mean_weekday_adjusted_index=("weekday_adjusted_index", "mean")).reset_index()
    named_events = events[events.event_name_1.notna()].groupby("event_name_1").agg(
        observed_days=("date", "size"), mean_daily_units=("units_sold", "mean"),
        mean_weekday_adjusted_index=("weekday_adjusted_index", "mean")
    ).reset_index()
    monthly = daily.assign(month_start=daily.date.dt.to_period("M").dt.to_timestamp()).groupby("month_start").agg(
        observed_days=("date", "size"), units_sold=("units_sold", "sum"), mean_daily_units=("units_sold", "mean")
    ).reset_index()
    weekly = daily.groupby("wm_yr_wk").agg(week_start=("date", "min"), observed_days=("date", "size"), units_sold=("units_sold", "sum"), mean_daily_units=("units_sold", "mean")).reset_index()
    subgroups = {}
    for name, frame, key in (("store", store_daily, "store_id"), ("category", category_daily, "cat_id")):
        frame = frame.copy()
        frame["weekday_number"] = pd.to_datetime(frame.date).dt.dayofweek
        summary = frame.groupby([key, "weekday_number"]).agg(mean_daily_units=("units_sold", "mean"), observed_days=("date", "size")).reset_index()
        summary["weekday_index"] = summary.mean_daily_units / summary[key].map(frame.groupby(key).units_sold.mean())
        subgroups[name + "_weekday"] = summary
    return {"daily": daily, "weekday": weekdays, "month_of_year": months, "weekday_by_year": stability,
            "monthly": monthly, "weekly": weekly, "event_associations": event_summary,
            "named_event_associations": named_events, **subgroups}


def select_series(series, config):
    eligible = series[series.units_sold.gt(0)].sort_values(["units_sold", "store_id", "item_id"], ascending=[False, True, True])
    if eligible.empty:
        eligible = series.sort_values(["store_id", "item_id"])
    choices = []
    if len(eligible):
        for rank, label in ((0, "high_volume"), (len(eligible) // 2, "middle_volume"), (len(eligible) - 1, "low_volume")):
            row = eligible.iloc[rank]
            choices.append({"store_id": row.store_id, "item_id": row.item_id, "selection": label})
        for regime in ("lumpy", "intermittent", "erratic", "smooth"):
            subset = eligible[eligible.regime.eq(regime)]
            if len(subset):
                row = subset.iloc[0]
                choices.append({"store_id": row.store_id, "item_id": row.item_id, "selection": "regime_" + regime})
    return pd.DataFrame(choices, columns=["store_id", "item_id", "selection"]).drop_duplicates(["store_id", "item_id"]).head(config.selected_series)


def decomposition_examples(daily, category_daily):
    from statsmodels.tsa.seasonal import STL

    selected = [("overall", daily)]
    selected.extend((str(key), values) for key, values in category_daily.groupby("cat_id"))
    results = []
    for label, frame in selected:
        frame = frame.sort_values("date")
        if len(frame) < 14:
            continue
        dates = pd.to_datetime(frame.date)
        if not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError("Decomposition requires continuous daily observations")
        fit = STL(frame.units_sold.to_numpy(dtype=float), period=7, robust=True).fit()
        results.append(pd.DataFrame({"series": label, "date": dates.to_numpy(), "units_sold": frame.units_sold.to_numpy(),
                                     "trend": fit.trend, "seasonal": fit.seasonal, "residual": fit.resid}))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["series", "date", "units_sold", "trend", "seasonal", "residual"])


def validate_totals(frames, baseline):
    """Additive quantities only; never sum distinct counts or variability ratios."""
    for name in ("store", "state", "category", "product", "store_item"):
        frame = frames[name]
        if int(frame.units_sold.sum()) != baseline["units_sold"]:
            raise ValueError(f"Unit reconciliation failed: {name}")
        value = sum((v for v in frame.known_revenue if pd.notna(v)), Decimal(0))
        if value != (baseline["known_revenue"] or Decimal(0)):
            raise ValueError(f"Revenue reconciliation failed: {name}")
        count_key = "observed_periods" if name == "store_item" else "observation_count"
        zero_key = "zero_periods" if name == "store_item" else "zero_sales_count"
        for key, expected in ((count_key, baseline["daily_rows"]), (zero_key, baseline["zero_sales_count"]),
                              ("missing_price_count", baseline["missing_price_count"]), ("unknown_revenue_count", baseline["unknown_revenue_count"])):
            if int(frame[key].sum()) != expected:
                raise ValueError(f"Count reconciliation failed: {name}.{key}")
    product = frames["product_segments"]
    if len(product) != len(frames["product"]) or product.item_id.duplicated().any():
        raise ValueError("ABC/XYZ fan-out")
    shares = product.revenue_share.dropna().sum()
    eligible_revenue = product.loc[product.abc_class.isin(list("ABC")), "known_revenue"].sum()
    if eligible_revenue and not math.isclose(shares, 1, abs_tol=1e-10):
        raise ValueError("ABC shares do not reconcile")
    cells = pd.concat([frames["abc_xyz_matrix"], frames["abc_xyz_unclassified"]], ignore_index=True)
    if int(cells.products.sum()) != len(product) or int(cells.units_sold.sum()) != baseline["units_sold"]:
        raise ValueError("ABC/XYZ matrix coverage does not reconcile")
    for summary in (cells, frames["regime_summary"]):
        if baseline["units_sold"] and not math.isclose(summary.volume_share.sum(), 1, abs_tol=1e-10):
            raise ValueError("Segment volume shares do not reconcile")
        if baseline["known_revenue"] and not math.isclose(summary.revenue_share.sum(), 1, abs_tol=1e-10):
            raise ValueError("Segment revenue shares do not reconcile")
