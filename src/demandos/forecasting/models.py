"""Small, bounded baseline library. Inputs contain ONE daily history, never targets."""

from dataclasses import dataclass, field
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


MODELS = ("naive", "seasonal_naive", "moving_average", "ses", "holt", "holt_damped",
          "holt_winters", "arima", "sarima", "croston", "sba", "tsb")


@dataclass
class ForecastResult:
    model: str
    status: str
    reason: str | None = None
    predictions: np.ndarray | None = None
    raw_predictions: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    runtime_seconds: float = 0.0


def history_values(history):
    """Require ordered, complete daily dates and finite, nonnegative observations."""
    if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("Expected one pd.Series with daily DatetimeIndex")
    dates = history.index
    if not len(history) or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Empty, missing, duplicate or unordered history dates")
    if not dates.equals(dates.normalize()) or not dates.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("History must have complete midnight daily observations")
    values = history.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("History must contain finite nonnegative observed sales")
    return values


def at_origin(frame, origin, lookback=None):
    """Clip BEFORE validation or fitting; reject mixed store/item identities."""
    for key in ("series_id", "store_id", "item_id"):
        if key in frame and frame[key].nunique(dropna=False) != 1:
            raise ValueError("Cannot mix series histories")
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or origin != origin.normalize():
        raise ValueError("Invalid forecast origin")
    prefix = frame.loc[pd.to_datetime(frame.date).le(origin), ["date", "units_sold"]].copy()
    history = pd.Series(prefix.units_sold.to_numpy(), index=pd.DatetimeIndex(prefix.date))
    history_values(history)
    if history.index[-1] != origin:
        raise ValueError("No observed sales at forecast origin")
    if lookback is not None:
        if type(lookback) is not int or lookback < 1:
            raise ValueError("lookback must be a positive integer")
        history = history.iloc[-lookback:]
    return history


def intermittent_state(values, alpha, beta, method):
    """First size=y[first positive]; interval=first index+1. No leading imputation."""
    first = int(np.flatnonzero(values > 0)[0])
    size, interval = float(values[first]), float(first + 1)
    probability, last = 1.0 / interval, first
    for index in range(first + 1, len(values)):
        positive = values[index] > 0
        if method == "tsb":
            probability += beta * (float(positive) - probability)
        if positive:
            size += alpha * (values[index] - size)
            interval += alpha * ((index - last) - interval)
            last = index
    rate = size * probability if method == "tsb" else size / interval
    if method == "sba":
        rate *= 1 - alpha / 2
    return rate, {"size": size, "interval": interval, "probability": probability,
                  "first_positive_index": first, "last_positive_index": last}


def _library_fit(model, values, m, params, maxiter):
    if model in ("arima", "sarima"):
        order = tuple(params.get("order", (1, 0, 0)))
        seasonal = tuple(params.get("seasonal_order", (0, 1, 1, m))) if model == "sarima" else (0, 0, 0, 0)
        if len(order) != 3 or len(seasonal) != 4 or any(type(n) is not int or n < 0 or n > 2 for n in (*order, *seasonal[:3])):
            raise ValueError("Orders must be small nonnegative integers <=2")
        if seasonal[3] not in (0, m) or (model == "sarima" and seasonal[3] != m):
            raise ValueError("Seasonal order must use configured seasonal period")
        fit = ARIMA(values, order=order, seasonal_order=seasonal).fit(
            method="statespace", method_kwargs={"maxiter": maxiter, "disp": 0})
        return fit, {"order": order, "seasonal_order": seasonal, "optimizer": "L-BFGS", "maxiter": maxiter,
                     "parameters": dict(zip(fit.param_names, fit.params.tolist()))}
    initialization = "known" if "initial_level" in params else "estimated"
    init = {"initialization_method": initialization}
    for name in ("initial_level", "initial_trend", "initial_seasonal"):
        if name in params:
            init[name] = params[name]
    if model == "ses":
        estimator = SimpleExpSmoothing(values, **init)
    elif model in ("holt", "holt_damped"):
        estimator = Holt(values, damped_trend=model == "holt_damped", **init)
    else:
        estimator = ExponentialSmoothing(values, trend="add", seasonal="add", seasonal_periods=m, **init)
    optimized = params.get("optimized", True)
    kwargs = {name: params[name] for name in ("smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend") if name in params}
    fit = estimator.fit(optimized=optimized, use_brute=False, method="L-BFGS-B",
                        minimize_kwargs={"options": {"maxiter": maxiter}}, **kwargs)
    fitted_parameters = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fit.params.items()}
    return fit, {"initialization": initialization, "optimized": optimized, "maxiter": maxiter,
                 "optimizer": "L-BFGS-B", "parameters": fitted_parameters}


def forecast(history, horizon, model, *, seasonal_period=7, params=None):
    """Return status plus full vector; failed/skipped fits have no predictions.

    Negative raw model outputs are retained and clipped to zero for scoring.
    Constant optimization histories are skipped, without hidden fallback models.
    """
    started = perf_counter()
    result = ForecastResult(model=model, status="failed")
    params = dict(params or {})
    caught = []
    try:
        if model not in MODELS or type(horizon) is not int or not 1 <= horizon <= 366:
            raise ValueError("Unknown model or horizon outside 1..366")
        if type(seasonal_period) is not int or not 2 <= seasonal_period <= 31:
            raise ValueError("Seasonal period must be an integer in 2..31")
        allowed = {"naive": set(), "seasonal_naive": set(), "moving_average": {"window"},
                   "croston": {"alpha"}, "sba": {"alpha"}, "tsb": {"alpha", "beta"},
                   "arima": {"order", "maxiter"}, "sarima": {"order", "seasonal_order", "maxiter"}}
        smoothing = {"initial_level", "optimized", "smoothing_level", "maxiter"}
        allowed.update(ses=smoothing, holt=smoothing | {"initial_trend", "smoothing_trend"},
                       holt_damped=smoothing | {"initial_trend", "smoothing_trend", "damping_trend"},
                       holt_winters=smoothing | {"initial_trend", "initial_seasonal", "smoothing_trend", "smoothing_seasonal"})
        if set(params) - allowed[model]:
            raise ValueError(f"Unsupported parameters: {sorted(set(params) - allowed[model])}")
        values = history_values(history)
        m = seasonal_period
        result.metadata = {"observations": len(values), "fit_start": history.index[0], "fit_end": history.index[-1],
                           "seasonal_period": m, "requested_parameters": params, "clipped_count": 0}
        if model == "naive":
            raw = np.repeat(values[-1], horizon)
        elif model == "seasonal_naive":
            if len(values) < m:
                result.status, result.reason = "skipped", "less_than_one_season"
                return result
            raw = np.resize(values[-m:], horizon)
        elif model == "moving_average":
            window = params.get("window", 28)
            if type(window) is not int or window < 1:
                raise ValueError("window must be a positive integer")
            if len(values) < window:
                result.status, result.reason = "skipped", "less_than_window"
                return result
            raw = np.repeat(np.mean(values[-window:]), horizon)
        elif model in ("croston", "sba", "tsb"):
            alpha, beta = params.get("alpha", .1), params.get("beta", .1)
            if not 0 < alpha <= 1 or not 0 < beta <= 1:
                raise ValueError("Smoothing alpha/beta must be in (0,1]")
            positives = np.count_nonzero(values)
            if positives == 0:
                raw = np.zeros(horizon)
                result.reason = "no_positive_history_zero_rate"
            elif positives < 2:
                result.status, result.reason = "skipped", "fewer_than_two_positive_events"
                return result
            else:
                rate, state = intermittent_state(values, alpha, beta, model)
                result.metadata.update(state)
                raw = np.repeat(rate, horizon)
        else:
            maxiter = params.get("maxiter", 100)
            if type(maxiter) is not int or not 1 <= maxiter <= 500:
                raise ValueError("maxiter must be an integer in 1..500")
            minimum = 2 * m + 1 if model in ("holt_winters", "sarima") else 1 if model == "ses" and "initial_level" in params else 3
            if len(values) < minimum:
                result.status, result.reason = "skipped", "insufficient_history"
                return result
            if np.ptp(values) == 0:
                result.status, result.reason = "skipped", "all_zero" if values[0] == 0 else "constant"
                return result
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit, metadata = _library_fit(model, values, m, params, maxiter)
                result.metadata.update(metadata)
                state = getattr(fit, "mle_retvals", None)
                if state is not None:
                    converged = state.get("converged", state.get("success", True))
                    result.metadata["converged"] = bool(converged)
                    result.metadata["iterations"] = state.get("iterations", state.get("nit"))
                    if not converged:
                        result.reason = "optimizer_did_not_converge"
                        return result
                raw = np.asarray(fit.forecast(horizon), dtype=float)
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (horizon,) or not np.isfinite(raw).all():
            raise ValueError("Model produced nonfinite or wrong-length predictions")
        result.raw_predictions = raw.copy()
        result.predictions = np.maximum(raw, 0)
        result.metadata["clipped_count"] = int((raw < 0).sum())
        result.status = "ok"
    except (ValueError, TypeError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
        result.reason = f"{type(error).__name__}: {error}"
    finally:
        result.warnings = [f"{type(w.message).__name__}: {w.message}" for w in caught]
        result.runtime_seconds = perf_counter() - started
    return result
