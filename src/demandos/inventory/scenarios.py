"""Bounded, origin-safe historical paths and explicitly assumed forecast CDFs."""
import hashlib

import numpy as np
import pandas as pd

from demandos.intermittent.diagnostics import prefix
from .core import counts, integer, Distribution


def seed_for(seed, *keys):
    return int.from_bytes(hashlib.sha256((str(seed)+"|"+"|".join(map(str, keys))).encode()).digest()[:8], "little")


def bootstrap(history, origin, *, samples=512, lookback=365, block=7, horizon=28, seed=42):
    """Weekday-aligned contiguous blocks, sampled with replacement from <=E.

    Blocks never wrap or cross the origin. Breaks dependence across blocks.
    block=1 is the separate weekday-conditional independent sensitivity.
    Randomized historical CDF ranks couple assumed forecast marginals across
    days; random tie breaking does not identify a true latent-demand copula.
    """
    samples, lookback, block, horizon = [integer(v, minimum=1) for v in (samples, lookback, block, horizon)]
    eligible = prefix(history, origin, lookback)
    if len(eligible) < 14 or block > len(eligible):
        raise ValueError("Insufficient eligible history for declared block bootstrap")
    y = eligible.units_sold.to_numpy(int)
    dates = pd.DatetimeIndex(eligible.date)
    future = pd.date_range(pd.Timestamp(origin)+pd.Timedelta(days=1), periods=horizon)
    rng = np.random.default_rng(seed)
    paths = np.empty((samples, horizon), dtype=int)
    ranks = np.empty((samples, horizon), dtype=float)
    for h in range(0, horizon, block):
        length = min(block, horizon-h)
        starts = np.flatnonzero((dates.dayofweek == future[h].dayofweek) & (np.arange(len(y))+length <= len(y)))
        if not len(starts):
            raise ValueError("No fully observed weekday-aligned historical block")
        drawn = rng.choice(starts, size=samples)
        for offset in range(length):
            values = y[drawn+offset]
            paths[:, h+offset] = values
            reference = np.sort(y[starts+offset])
            left = np.searchsorted(reference, values, side="left")
            right = np.searchsorted(reference, values, side="right")
            ranks[:, h+offset] = (left+rng.random(samples)*(right-left))/len(reference)
    return paths, ranks, dict(history_start=dates.min(), history_end=dates.max(), history_rows=len(y),
                              historical_zero_rate=float(np.mean(y == 0)), block_days=block,
                              samples=samples, seed=str(seed), status="SIMULATED_SALES_AS_DEMAND")


def quantile_paths(quantiles, ranks, tail_floor=1., width_factor=None):
    """ASSUMED piecewise-linear inverse CDF: knots (0,.1,.5,.9,1).

    Tail anchors 0 and q90+max(q90-q50, tail_floor) are chosen assumptions.
    The full CDF and rank coupling are NOT identified by P10/P50/P90.
    Half-up rounding creates integer demand and may shift nominal quantiles.
    Optional interval-width inflation is a heuristic around the SAME base
    median; a conformal upper endpoint is NEVER assigned quantile probability.
    """
    q, u = np.asarray(quantiles, float), np.asarray(ranks, float)
    if q.ndim != 2 or q.shape[1] != 3 or u.ndim != 2 or u.shape[1] != len(q):
        raise ValueError("Require aligned horizon x 3 quantiles and sample x horizon ranks")
    if not np.isfinite(q).all() or (q < 0).any() or (np.diff(q, axis=1) < 0).any() or not np.isfinite(u).all() or (u < 0).any() or (u > 1).any():
        raise ValueError("Invalid repaired quantiles or CDF ranks")
    if not np.isfinite(tail_floor) or tail_floor <= 0:
        raise ValueError("Positive finite tail floor required")
    factor = np.ones(len(q)) if width_factor is None else np.asarray(width_factor, float)
    if factor.shape != (len(q),) or not np.isfinite(factor).all() or (factor < 1).any():
        raise ValueError("Aligned finite noncontracting width factors required")
    paths = np.empty(u.shape)
    for h, (lower, median, upper) in enumerate(q):
        anchors = [0., lower, median, upper, upper+max(upper-median, tail_floor)]
        raw = np.interp(u[:, h], [0., .1, .5, .9, 1.], anchors)
        paths[:, h] = np.maximum(0., median+(raw-median)*factor[h])
    return np.floor(paths+.5).astype(np.int64)


def period_distribution(paths, leads, probabilities, review=0):
    y = counts(paths)
    if y.ndim != 2:
        raise ValueError("Coherent daily paths required")
    review = integer(review)
    Distribution(leads, probabilities)
    if len(set(leads)) != len(leads):
        raise ValueError("Distinct lead supports required")
    values, weights = [], []
    for lead, probability in zip(leads, probabilities):
        lead = integer(lead, "lead", 1)
        if lead+review > y.shape[1]:
            raise ValueError("Incomplete protection window; explicit skip required")
        values.extend(y[:, :lead+review].sum(axis=1))
        weights.extend(np.full(len(y), probability/len(y)))
    return Distribution(values, weights)


def targets(frame, origin, horizon=28):
    """Verify daily alignment including 2016-02-29; never silently truncate."""
    rows = frame.sort_values("target_date")
    expected = pd.date_range(pd.Timestamp(origin)+pd.Timedelta(days=1), periods=horizon)
    if len(rows) != horizon or not pd.DatetimeIndex(rows.target_date).equals(expected):
        raise ValueError("Missing/duplicate/misaligned complete target window")
    if "horizon" in rows and not np.array_equal(rows.horizon, np.arange(1, horizon+1)):
        raise ValueError("Forecast horizon and date disagree")
    return rows.reset_index(drop=True)
