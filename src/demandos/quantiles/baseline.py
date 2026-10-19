"""Unconditional trailing empirical inverse-CDF quantiles, including observed zeros."""

import numpy as np
import pandas as pd
from .contracts import LEVELS
from demandos.intermittent.diagnostics import prefix


def historical(history, origin, days=56):
    if type(days) is not int or days < 1:
        raise ValueError("Positive historical baseline window required")
    observed = prefix(history, origin, visible_days=days)
    if observed.empty:
        raise ValueError("Historical quantiles unavailable: no observed prefix; no invented zeros")
    y = observed.units_sold.to_numpy(float)
    quantiles = np.quantile(y, LEVELS, method="inverted_cdf")
    status = "all_zero" if not y.any() else "short_history" if len(y) < days else "complete"
    return np.tile(quantiles, (28, 1)), {"status": status, "history_days": len(y),
        "history_start": observed.date.min(), "history_end": observed.date.max(),
        "fit_cutoff": pd.Timestamp(origin), "fit_target_max": observed.date.max(),
        "zero_fraction": float((y == 0).mean()), "quantile_method": "inverted_cdf_including_zeros"}


def inverse_cdf(values, probabilities, tau):
    x, p = np.asarray(values, float), np.asarray(probabilities, float)
    if (x.ndim != 1 or x.shape != p.shape or not len(x) or not np.isfinite(x).all()
            or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1)
            or not 0 < tau < 1 or (np.diff(x) < 0).any()):
        raise ValueError("Ordered finite support and valid discrete probability mass required")
    return float(x[min(np.searchsorted(np.cumsum(p), tau, side="left"), len(x)-1)])


def nonadditivity():
    single = inverse_cdf([0, 10], [.9, .1], .9)
    total = inverse_cdf([0, 10, 20], [.81, .18, .01], .9)
    return {"distribution": "Independent A,B: P(0)=0.9, P(10)=0.1", "quantile_definition": "inverse CDF",
            "q90_A": single, "q90_B": single, "sum_marginal_q90": 2*single, "q90_sum": total,
            "conclusion": "Marginal quantiles are not additive; no probabilistic reconciliation is claimed"}
