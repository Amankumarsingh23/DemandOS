"""Observed history is distinct from missing or deliberately withheld history."""
import numpy as np
import pandas as pd
from .config import Config


def prefix(history, origin, visible_days=None):
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or origin != origin.normalize():
        raise ValueError("Daily origin required")
    for name in ("store_id", "item_id"):
        if name in history and history[name].nunique(dropna=False) > 1:
            raise ValueError("Cannot mix store-item histories")
    if visible_days is not None and (type(visible_days) is not int or visible_days < 0):
        raise ValueError("Visible-history restriction must be a nonnegative day count")
    result = history.loc[pd.to_datetime(history.date).le(origin), ["date", "units_sold"]].copy()
    if visible_days is not None:
        result = result.loc[pd.to_datetime(result.date).gt(origin-pd.Timedelta(days=visible_days))]
    result["date"] = pd.to_datetime(result.date)
    if not result.empty:
        dates = pd.DatetimeIndex(result.date)
        values = result.units_sold.to_numpy(float)
        if dates.hasnans or dates.has_duplicates or not dates.equals(pd.date_range(dates[0], origin)):
            raise ValueError("Visible observations must be ordered, complete and end at origin")
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Observed sales must be finite nonnegative integers")
    return result.reset_index(drop=True)


def quadrant(adi, cv2, config=Config()):
    if adi is None or cv2 is None or not np.isfinite([adi,cv2]).all() or adi < 1 or cv2 < 0:
        raise ValueError("Finite valid intermittency statistics required")
    sparse, variable = adi >= config.adi_boundary, cv2 >= config.cv2_boundary
    return ("lumpy" if variable else "intermittent") if sparse else ("erratic" if variable else "smooth")


def statistics(history, origin, config=Config(), window=None):
    observed = prefix(history, origin)
    if window is not None:
        observed = observed.loc[observed.date.gt(pd.Timestamp(origin)-pd.Timedelta(days=window))]
    y = observed.units_sold.to_numpy(float)
    positive = y[y > 0]
    n, k = len(y), len(positive)
    mean = float(positive.mean()) if k else None
    variance = float(positive.var(ddof=0)) if k else None
    cv2 = variance/(mean*mean) if k >= 2 else None
    status = "no_observations" if n == 0 else "all_zero" if k == 0 else "insufficient_positive_history" if k < 2 else "classified"
    return {"observed_history_days": n, "observed_positive_days": k,
            "zero_sales_fraction": float(np.mean(y == 0)) if n else None,
            "positive_occurrence_fraction": k/n if n else None,
            "positive_sales_mean": mean, "positive_sales_population_variance": variance,
            "adi": n/k if k else None, "cv2": cv2, "diagnostic_status": status,
            "regime": quadrant(n/k,cv2,config) if status == "classified" else status,
            "days_since_positive": int((pd.Timestamp(origin)-observed.loc[observed.units_sold.gt(0), "date"].max()).days) if k else None}


def cold_status(observed, *, known_category=True, item_elsewhere=False, short_history=28):
    n = len(observed)
    if not known_category:
        return "unknown_category"
    if n == 0:
        return "new_store_existing_item" if item_elsewhere else "no_observed_history"
    if n < 7:
        return "short_under_7"
    if n < short_history:
        return "short_under_28"
    if not observed.units_sold.gt(0).any():
        return "observed_all_zero"
    return "established"


def describe(history, origin, config=Config(), *, visible_days=None, known_category=True, item_elsewhere=False):
    observed = prefix(history, origin, visible_days)
    result = statistics(observed, origin, config, config.history_window)
    trailing = statistics(observed, origin, config, config.trailing_window)
    result.update(visible_history_days=len(observed), visible_history_start=observed.date.min() if len(observed) else pd.NaT,
                  history_is_restricted=visible_days is not None,
                  cold_status=cold_status(observed,known_category=known_category,item_elsewhere=item_elsewhere,short_history=config.short_history),
                  trailing_regime=trailing["regime"], trailing_adi=trailing["adi"], trailing_cv2=trailing["cv2"])
    return result
