"""Exact discrete upper-tail risk, including fractional boundary mass."""
import numpy as np
from scipy.optimize import linprog


def distribution(losses, probabilities):
    x, p = np.asarray(losses, float), np.asarray(probabilities, float)
    if x.ndim != 1 or not len(x) or p.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("Finite aligned losses and normalized nonnegative probabilities required")
    return x, p


def tail(losses, probabilities, alpha=.9):
    x, p = distribution(losses, probabilities)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Confidence level must be strictly between zero and one")
    positive = p > 0
    order = np.argsort(x[positive], kind="stable")
    values, weights = x[positive][order], p[positive][order]
    index = min(np.searchsorted(np.cumsum(weights), np.nextafter(alpha,-np.inf), side="left"), len(values)-1)
    remaining, total = 1-alpha, 0.
    for value, weight in zip(values[::-1], weights[::-1]):
        taken = min(remaining, weight)
        total += taken*value
        remaining -= taken
        if remaining <= 0:
            break
    return dict(expected=float(x@p), var=float(values[index]), cvar=float(total/(1-alpha)),
                worst=float(values[-1]), alpha=float(alpha))


def epigraph_cvar(losses, probabilities, alpha=.9):
    """Independent fixed-loss LP; eta is explicitly unrestricted, including negatives."""
    x, p = distribution(losses, probabilities)
    tail(x, p, alpha)
    n = len(x)
    result = linprog(np.r_[1., p/(1-alpha)], A_ub=np.c_[-np.ones(n), -np.eye(n)],
                     b_ub=-x, bounds=[(None, None)]+[(0, None)]*n, method="highs")
    if not result.success:
        raise AssertionError(result.message)
    return dict(cvar=float(result.fun), eta=float(result.x[0]), excess=result.x[1:].tolist())


def worst_probability(normal_cost, shock_cost, interval=(.05, .20)):
    a, b = interval
    if not np.isfinite([normal_cost, shock_cost, a, b]).all() or not 0 <= a <= b <= 1:
        raise ValueError("Invalid two-family probability interval")
    values = [(1-p)*normal_cost+p*shock_cost for p in interval]
    index = int(np.argmax(values))  # stable first endpoint in a tie
    return dict(cost=float(values[index]), shock_probability=float(interval[index]))


def frontier(frame, mean="expected", risk="cvar"):
    """Nondominated points among the tested integer allocations only."""
    a = frame[[mean, risk]].to_numpy(float)
    if not np.isfinite(a).all():
        raise ValueError("Finite frontier coordinates required")
    return [not any(np.all(other <= point+1e-7) and np.any(other < point-1e-7) for other in a) for point in a]
