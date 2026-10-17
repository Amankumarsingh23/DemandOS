"""Daily-panel validation, transformations and auditable information boundaries."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


KEYS = ["store_id", "item_id"]


def daily_panel(frame):
    """Canonical ordering; reject duplicate/gapped/nonfinite observations, never impute."""
    required = KEYS + ["date", "units_sold"]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError("Expected a nonempty store/item/date/units panel")
    result = frame.copy(deep=True)
    if result[required].isna().any().any():
        raise ValueError("Missing identifiers, dates or observations")
    result["date"] = pd.to_datetime(result.date, errors="raise")
    if not result.date.eq(result.date.dt.normalize()).all():
        raise ValueError("Expected midnight daily dates")
    result["units_sold"] = pd.to_numeric(result.units_sold, errors="raise")
    if not np.isfinite(result.units_sold).all() or result.units_sold.lt(0).any():
        raise ValueError("Observed sales must be finite and nonnegative")
    if result.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate daily series keys")
    result = result.sort_values(KEYS + ["date"]).reset_index(drop=True)
    differences = result.groupby(KEYS).date.diff().dropna()
    if not differences.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Missing daily dates; observations are not imputed")
    return result


def descriptive_lags(frame, lags=(1, 7, 14, 28), window=28):
    """Target-date historical alignment, NOT automatically valid for h>1 forecasts."""
    if window < 1 or any(not isinstance(lag, int) or lag < 1 for lag in lags):
        raise ValueError("Positive integer lags/window required")
    result = daily_panel(frame)
    groups = result.groupby(KEYS).units_sold
    for lag in lags:
        result[f"target_lag_{lag}"] = groups.shift(lag)
    result["prior_mean"] = groups.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    result["prior_variance"] = groups.transform(lambda s: s.shift(1).rolling(window, min_periods=1).var(ddof=0))
    return result


def difference(values, lag=1, order=1):
    if not isinstance(lag, int) or lag < 1 or order not in (1, 2):
        raise ValueError("Positive lag and order 1 or 2 required")
    result = pd.Series(values, copy=True).astype(float)
    for _ in range(order):
        result = result.diff(lag)
    return result


@dataclass(frozen=True)
class FeatureEvidence:
    name: str
    value: object
    source_date: object
    available_at: object
    kind: str = "sales"
    fitted_through: object = None
    centered: bool = False
    descriptive_only: bool = False
    availability_verified: bool = False


def validate_feature(evidence, origin):
    """Check declared provenance; cannot establish unknown real-world availability."""
    origin = pd.Timestamp(origin)
    if pd.isna(origin):
        raise ValueError("Forecast origin is required")
    source, available = pd.Timestamp(evidence.source_date), pd.Timestamp(evidence.available_at)
    if pd.isna(source) or pd.isna(available):
        raise ValueError("Feature source and availability dates are required")
    if evidence.centered or evidence.descriptive_only:
        raise ValueError("Centered/descriptive components are forbidden as forecast features")
    if available > origin:
        raise ValueError("Feature becomes available after forecast origin")
    if evidence.fitted_through is not None:
        if pd.isna(pd.Timestamp(evidence.fitted_through)) or pd.Timestamp(evidence.fitted_through) > origin:
            raise ValueError("Transformation fit cutoff is unknown or after forecast origin")
    if evidence.kind == "sales" and source > origin:
        raise ValueError("Future actual sales leak past forecast origin")
    if evidence.kind == "price" and not evidence.availability_verified:
        raise ValueError("Retrospective prices have no verified as-of availability")
    if evidence.kind not in ("sales", "calendar", "price"):
        raise ValueError("Unknown feature provenance kind")
    return True


def fit_location_scale(frame, fit_end):
    data = daily_panel(frame)
    fit_end = pd.Timestamp(fit_end)
    prefix = data[data.date.le(fit_end)]
    if prefix.empty:
        raise ValueError("No observations available for fitting")
    fitted = prefix.groupby(KEYS).units_sold.agg(mean="mean", std=lambda s: s.std(ddof=0)).reset_index()
    fitted["fitted_through"] = prefix.groupby(KEYS).date.max().to_numpy()
    return fitted


def origin_features(frame, origin, horizons=(1, 7, 14, 28), lags=(1, 7, 14, 28), window=28, fitted=None):
    """Direct-horizon inputs. origin_minus_k means y_(origin-k), independent of h."""
    if window < 1 or any(not isinstance(h, int) or h < 1 for h in (*horizons, *lags)):
        raise ValueError("Positive integer horizons, lags and window required")
    data = daily_panel(frame)
    origin = pd.Timestamp(origin)
    # The prefix is selected before any data-derived calculations.
    data = data[data.date.le(origin)]
    rows = []
    for key, history in data.groupby(KEYS, sort=True):
        values = history.set_index("date").units_sold
        if origin not in values.index:
            raise ValueError("Every selected series must contain the forecast origin")
        base = dict(zip(KEYS, key))
        base.update(origin=origin, sales_at_origin=float(values.loc[origin]), max_sales_source_date=origin,
                    origin_rolling_mean=float(values.tail(window).mean()),
                    origin_rolling_variance=float(values.tail(window).var(ddof=0)),
                    rolling_observations=min(window, len(values)))
        evidence = [FeatureEvidence("sales_at_origin", base["sales_at_origin"], origin, origin)]
        for lag in lags:
            source = origin - pd.Timedelta(days=lag)
            base[f"origin_minus_{lag}"] = float(values.loc[source]) if source in values.index else None
            base[f"origin_minus_{lag}_source_date"] = source if source in values.index else pd.NaT
            if source in values.index:
                evidence.append(FeatureEvidence(f"origin_minus_{lag}", base[f"origin_minus_{lag}"], source, source))
        if fitted is not None:
            match = fitted[(fitted.store_id == key[0]) & (fitted.item_id == key[1])]
            if len(match) != 1:
                raise ValueError("One fitted parameter row required per series")
            parameters = match.iloc[0]
            evidence.append(FeatureEvidence("fitted_scale", parameters["mean"], parameters.fitted_through,
                                            parameters.fitted_through, fitted_through=parameters.fitted_through))
            base["scale_fit_end"] = parameters.fitted_through
            base["scaled_origin_sales"] = (base["sales_at_origin"] - parameters["mean"]) / parameters["std"] if parameters["std"] > 0 else None
        for feature in evidence:
            validate_feature(feature, origin)
        for horizon in horizons:
            target = origin + pd.Timedelta(days=horizon)
            validate_feature(FeatureEvidence("target_weekday", target.dayofweek, target, origin, kind="calendar"), origin)
            rows.append(dict(base, horizon=horizon, target_date=target, target_weekday=target.dayofweek))
    if not rows:
        raise ValueError("No histories available at origin")
    return pd.DataFrame(rows)


def chronological_split(dates, validation_days=28, test_days=28, minimum_train_days=84):
    index = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    if index.empty or index.has_duplicates or not index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Unique continuous daily calendar required")
    if min(validation_days, test_days, minimum_train_days) < 1 or len(index) < validation_days + test_days + minimum_train_days:
        raise ValueError("Insufficient dates for chronological split")
    return {"train_start": index[0], "train_end": index[-validation_days-test_days-1],
            "validation_start": index[-validation_days-test_days], "validation_end": index[-test_days-1],
            "test_start": index[-test_days], "test_end": index[-1]}


def origin_schedule(dates, initial_train_days, horizon=28, step=28, training_window=None):
    index = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    if index.has_duplicates or not index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Unique continuous daily calendar required")
    if min(initial_train_days, horizon, step) < 1 or (training_window is not None and training_window < 1):
        raise ValueError("Positive origin/window lengths required")
    rows = []
    for end in range(initial_train_days - 1, len(index) - horizon, step):
        begin = 0 if training_window is None else max(0, end - training_window + 1)
        rows.append({"history_start": index[begin], "origin": index[end],
                     "target_start": index[end + 1], "target_end": index[end + horizon], "horizon": horizon})
    return pd.DataFrame(rows, columns=["history_start", "origin", "target_start", "target_end", "horizon"])
