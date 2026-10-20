"""Forecast-label maturity is checked before any score is inspected or selected."""

import numpy as np
import pandas as pd
from .core import scores

KEYS = ["store_id", "item_id", "origin_date", "horizon", "target_date"]
GROUP = ["family", "cohort", "horizon_band"]
ORDER = ["observation_available_date", "target_date", "origin_date", "store_id", "item_id", "horizon"]


def validate(frame):
    required = KEYS+["family", "model_id", "model_fit_cutoff", "fit_target_max", "selection_cutoff",
                     "observation_available_date", "cohort", "horizon_band", "q10", "q50", "q90", "actual", "score_id"]
    if frame.empty or not set(required).issubset(frame) or frame[required].isna().any().any():
        raise ValueError("Complete held-out score/forecast provenance required")
    if frame.duplicated(["family", *KEYS]).any() or frame.score_id.duplicated().any():
        raise ValueError("Duplicate held-out forecast keys or score IDs")
    for name in ["origin_date", "target_date", "model_fit_cutoff", "fit_target_max", "selection_cutoff", "observation_available_date"]:
        dates = pd.to_datetime(frame[name])
        if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
            raise ValueError("Daily provenance dates required")
    issue, target = pd.to_datetime(frame.origin_date), pd.to_datetime(frame.target_date)
    fit, used, selected = [pd.to_datetime(frame[k]) for k in ["model_fit_cutoff", "fit_target_max", "selection_cutoff"]]
    available = pd.to_datetime(frame.observation_available_date)
    if (not frame.horizon.isin(range(1, 29)).all() or not target.eq(issue+pd.to_timedelta(frame.horizon, unit="D")).all()
            or not used.le(fit).all() or not fit.le(issue).all() or not fit.lt(target).all()
            or not selected.le(issue).all() or not selected.lt(target).all() or not available.ge(target).all()):
        raise ValueError("In-sample, selection-contaminated or invalid maturity/date provenance")
    y = frame.actual.to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any() or (y != np.floor(y)).any():
        raise ValueError("Finite nonnegative integer observed sales required")
    if not np.isfinite(frame[["q10", "q50", "q90"]].to_numpy(float)).all() or frame.q10.lt(0).any() or frame.q10.gt(frame.q50).any() or frame.q50.gt(frame.q90).any():
        raise ValueError("Invalid held-out base quantiles")
    return frame


def scored(frame):
    validate(frame)
    result = frame.copy()
    for name, values in scores(result.actual, result.q10, result.q90, result.q50).items():
        result[name] = values
    return result


def eligible(ledger, asof):
    asof = pd.Timestamp(asof)
    if pd.isna(asof) or asof != asof.normalize():
        raise ValueError("Explicit daily score-availability cutoff required")
    # Invalid or missing FUTURE outcomes cannot influence an earlier calibrator.
    mask = (pd.to_datetime(ledger.target_date).le(asof)
            & pd.to_datetime(ledger.observation_available_date).le(asof)
            & pd.to_datetime(ledger.origin_date).le(asof))
    result = ledger.loc[mask].copy()
    return scored(result).sort_values(ORDER).reset_index(drop=True) if len(result) else result


def split_pool(ledger, asof, config):
    result = eligible(ledger, min(pd.Timestamp(asof), pd.Timestamp(config["split_end"])))
    return result[result.origin_date.isin(pd.to_datetime(config["split_origins"]))].copy()


def recent_pool(ledger, asof, count):
    if type(count) is not int or count < 1:
        raise ValueError("Positive bounded score-window size required")
    result = eligible(ledger, asof)
    return result.groupby(GROUP, sort=True, group_keys=False).tail(count) if len(result) else result


def complete_paths(pool, minimum_origins=3, minimum_paths=10):
    keys = ["family", "cohort", "store_id", "item_id", "origin_date"]
    records = []
    for key, group in pool.groupby(keys, sort=True):
        if len(group) == 28 and sorted(group.horizon.tolist()) == list(range(1, 29)):
            records.append({**dict(zip(keys, key)), "score": float(group.cqr_signed.max()),
                            "maturity_date": group.observation_available_date.max()})
    paths = pd.DataFrame(records, columns=[*keys, "score", "maturity_date"])
    statuses = []
    for key, group in paths.groupby(["family", "cohort"], sort=True):
        n, origins = len(group), group.origin_date.nunique()
        statuses.append({"family": key[0], "cohort": key[1], "n": n, "origin_blocks": origins,
                         "status": "available" if n >= minimum_paths and origins >= minimum_origins else "insufficient_complete_trajectories"})
    return paths, pd.DataFrame(statuses, columns=["family", "cohort", "n", "origin_blocks", "status"])
