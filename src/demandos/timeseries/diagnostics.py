"""Guarded library diagnostics, with explicit skipped/failed/inconclusive states."""

from dataclasses import dataclass
from inspect import signature
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.seasonal import STL


@dataclass(frozen=True)
class DiagnosticsConfig:
    validation_days: int = 28
    test_days: int = 28
    max_store_items: int = 14
    store_aggregates: int = 1
    category_aggregates: int = 3
    max_lags: int = 56
    minimum_observations: int = 84
    minimum_nonzero: int = 20
    minimum_nonzero_fraction: float = 0.1
    rolling_days: int = 28
    seasonal_period: int = 7
    include_second_difference: bool = True
    adf_maxlag: int = 14
    regression: str = "c"
    alpha: float = 0.05
    horizons: tuple = (1, 7, 14, 28)
    lags: tuple = (1, 7, 14, 28)

    def __post_init__(self):
        for name in ("validation_days", "test_days", "max_store_items", "store_aggregates",
                     "category_aggregates", "max_lags", "minimum_observations", "minimum_nonzero",
                     "rolling_days", "seasonal_period"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"Positive integer required: {name}")
        if self.max_store_items > 30 or self.store_aggregates > 10 or self.category_aggregates > 3:
            raise ValueError("Diagnostic cohort exceeds configured small-cohort limits")
        if self.minimum_observations < 16 or self.seasonal_period < 2 or self.adf_maxlag < 0:
            raise ValueError("Invalid diagnostic history/lag settings")
        if self.regression not in ("c", "ct") or not 0 < self.alpha < 1 or not 0 <= self.minimum_nonzero_fraction <= 1:
            raise ValueError("Invalid regression, alpha, or sparsity choice")
        if any(not isinstance(n, int) or n < 1 for n in (*self.horizons, *self.lags)):
            raise ValueError("Positive horizons and lags required")
        if not {1, 7, 14, 28}.issubset(self.horizons):
            raise ValueError("The demonstration must include horizons 1, 7, 14 and 28")


def library_result(function, values, **kwargs):
    # statsmodels 0.15 introduces result objects; explicitly retain the tested
    # tuple/array convention without passing an unknown argument to 0.14.
    if "result_object" in signature(function).parameters:
        kwargs["result_object"] = False
    return function(values, **kwargs)


def eligibility(values, config=DiagnosticsConfig()):
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not len(data):
        return "empty_or_not_univariate"
    if not np.isfinite(data).all():
        return "missing_or_nonfinite"
    if np.count_nonzero(data) == 0:
        return "all_zero"
    if np.ptp(data) <= max(1, np.max(np.abs(data))) * 1e-12:
        return "constant"
    if len(data) < config.minimum_observations:
        return "insufficient_history"
    if np.count_nonzero(data) < config.minimum_nonzero or np.count_nonzero(data) / len(data) < config.minimum_nonzero_fraction:
        return "extremely_sparse"
    return None


def correlation_diagnostics(values, config=DiagnosticsConfig()):
    reason = eligibility(values, config)
    if reason:
        return {"status": "skipped", "reason": reason, "coefficients": []}
    values = np.asarray(values, dtype=float)
    nlags = min(config.max_lags, len(values) // 2 - 1)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            a, aci = library_result(acf, values, nlags=nlags, adjusted=False, fft=True, alpha=config.alpha, bartlett_confint=True)
            p, pci = library_result(pacf, values, nlags=nlags, method="ywm", alpha=config.alpha)
        if not all(np.isfinite(array).all() for array in (a, aci, p, pci)):
            raise ValueError("Nonfinite correlation estimates")
        records = [{"lag": lag, "acf": float(a[lag]), "acf_low": float(aci[lag, 0]), "acf_high": float(aci[lag, 1]),
                    "pacf": float(p[lag]), "pacf_low": float(pci[lag, 0]), "pacf_high": float(pci[lag, 1])}
                   for lag in range(nlags + 1)]
        return {"status": "ok", "reason": None, "coefficients": records,
                "warnings": [str(w.message) for w in caught], "max_lags": nlags,
                "estimator": "ACF adjusted=False FFT Bartlett intervals; PACF Yule-Walker MLE (ywm)"}
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as error:
        return {"status": "failed", "reason": str(error), "coefficients": []}


def stationarity_diagnostics(values, config=DiagnosticsConfig()):
    reason = eligibility(values, config)
    if reason:
        return {"status": "skipped", "reason": reason, "interpretation": "skipped", "adf": None, "kpss": None}
    values = np.asarray(values, dtype=float)
    results = {}
    for name in ("adf", "kpss"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if name == "adf":
                    bound = len(values) // 2 - (2 if config.regression == "ct" else 1) - 1
                    outcome = library_result(adfuller, values, maxlag=min(config.adf_maxlag, bound), regression=config.regression, autolag="AIC")
                else:
                    outcome = library_result(kpss, values, regression=config.regression, nlags="auto")
            statistic, pvalue, used_lags = outcome[:3]
            if not np.isfinite([statistic, pvalue]).all():
                raise ValueError("Nonfinite test results")
            results[name] = {"status": "ok", "statistic": float(statistic), "pvalue": float(pvalue),
                             "lags": int(used_lags), "regression": config.regression,
                             "warnings": [str(w.message) for w in caught]}
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as error:
            results[name] = {"status": "failed", "reason": str(error), "pvalue": None}
    if any(result["status"] != "ok" for result in results.values()):
        return dict(status="partial_or_failed", reason="test_failure", interpretation="inconclusive", **results)
    a = results["adf"]["pvalue"] < config.alpha
    k = results["kpss"]["pvalue"] < config.alpha
    interpretation = ("conflicting" if k else "compatible_with_stationarity") if a else ("compatible_with_nonstationarity" if k else "inconclusive")
    return dict(status="ok", reason=None, interpretation=interpretation, **results)


def additive_stl(frame, period=7, usage="descriptive", origin=None):
    if usage != "descriptive":
        raise ValueError("STL output is descriptive-only; cannot export smoothed forecasting features")
    data = frame.sort_values("date").copy()
    data["date"] = pd.to_datetime(data.date)
    if origin is not None:
        data = data[data.date.le(pd.Timestamp(origin))].copy()
    if data.date.duplicated().any() or not data.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("STL requires unique continuous daily dates")
    if len(data) < 2 * period or not np.isfinite(data.units_sold).all() or np.ptp(data.units_sold) == 0:
        return pd.DataFrame(), "insufficient_nonconstant_regular_history"
    fit = STL(data.units_sold.to_numpy(dtype=float), period=period, robust=True).fit()
    data["trend"], data["seasonal"], data["residual"] = fit.trend, fit.seasonal, fit.resid
    data["fitted_through"] = data.date.max()
    data["usage"] = "descriptive_only"
    return data, None
