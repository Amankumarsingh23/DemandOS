"""Predeclared cohort/band calibration and delayed-feedback sequential ACI."""

from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd

from .core import corrected, weighted, intervals, update_alpha
from .ledger import KEYS, GROUP, split_pool, recent_pool, complete_paths


def audit(pool):
    ids = pool.score_id.astype("int64").tolist() if len(pool) else []
    return {"score_ids": ids, "score_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
            "calibration_target_min": pool.target_date.min() if len(pool) else pd.NaT,
            "calibration_target_max": pool.target_date.max() if len(pool) else pd.NaT,
            "availability_max": pool.observation_available_date.max() if len(pool) else pd.NaT,
            "calibration_issue_min": pool.origin_date.min() if len(pool) else pd.NaT,
            "calibration_issue_max": pool.origin_date.max() if len(pool) else pd.NaT,
            "origin_blocks": pool.origin_date.nunique() if len(pool) else 0,
            "distinct_target_dates": pool.target_date.nunique() if len(pool) else 0}


def make_rows(base, correction, name, info, *, point=False):
    lo, hi = (base.q50, base.q50) if point else (base.q10, base.q90)
    lower, upper, effective = intervals(lo, hi, correction)
    columns = [*KEYS, "family", "model_id", "cohort", "horizon_band", "score_id"]
    output = base[columns].copy()
    output["method"] = base.family.iloc[0]+"_"+name
    output["base_lower"], output["base_upper"] = np.asarray(lo), np.asarray(hi)
    output["lower"], output["upper"] = lower, upper
    output["correction"], output["effective_correction"] = correction, effective
    output["empty_set_enlarged"] = effective > correction
    output["calibration_n"], output["calibration_rank"] = info.get("n", 0), info.get("rank")
    output["calibration_alpha"] = info.get("alpha", .2)
    output["status"] = info.get("status", "base")
    return output


def static_intervals(ledger, origins, config):
    frames, thresholds, availability = [], [], []
    for asof in pd.to_datetime(origins):
        fixed, recent = split_pool(ledger, asof, config), recent_pool(ledger, asof, config["rolling_scores"])
        grouped_fixed = dict(tuple(fixed.groupby(GROUP, sort=True))) if len(fixed) else {}
        grouped_recent = dict(tuple(recent.groupby(GROUP, sort=True))) if len(recent) else {}
        empty = fixed.iloc[:0]
        forecast = ledger[ledger.origin_date.eq(asof)]
        for key, base in forecast.groupby(GROUP, sort=True):
            f, r = grouped_fixed.get(key, empty), grouped_recent.get(key, empty)
            options = [("base_repaired", None, None), ("split_absolute", f, "absolute"),
                       ("split_cqr", f, "cqr_signed"), ("split_cqr_nonnegative", f, "cqr_nonnegative"),
                       ("rolling_cqr", r, "cqr_signed"), ("weighted_cqr_heuristic", r, "cqr_signed")]
            for method, source, column in options:
                if source is None:
                    c, info = 0., {"n": 0, "rank": None, "alpha": config["alpha"], "status": "base", "heuristic": False}
                    metadata = {}
                elif method == "weighted_cqr_heuristic" and len(source):
                    age = (asof-pd.to_datetime(source.observation_available_date)).dt.days.to_numpy()
                    weights = np.exp2(-age/config["half_life_days"])
                    c, info = weighted(source[column], weights, config["alpha"])
                    metadata = {**audit(source), "normalized_weights": (weights/weights.sum()).tolist(),
                                "half_life_days": config["half_life_days"]}
                else:
                    c, info = corrected(source[column].to_numpy() if len(source) else [], config["alpha"])
                    metadata = audit(source)
                frames.append(make_rows(base, c, method, info, point=method == "split_absolute"))
                thresholds.append(dict(zip(GROUP, key), origin_date=asof, method=key[0]+"_"+method,
                                       correction=c, **info, **metadata))
        paths, statuses = complete_paths(fixed, config["path_min_origins"], config["path_min_trajectories"])
        for key, base in forecast.groupby(["family", "cohort"], sort=True):
            selected = statuses[statuses.family.eq(key[0]) & statuses.cohort.eq(key[1])]
            status = selected.iloc[0].to_dict() if len(selected) else {"status": "insufficient_complete_trajectories", "n": 0, "origin_blocks": 0}
            availability.append({"origin_date": asof, "family": key[0], "cohort": key[1], **status})
            if status["status"] != "available":
                continue
            sample = paths[paths.family.eq(key[0]) & paths.cohort.eq(key[1])]
            c, info = corrected(sample.score, config["alpha"])
            frames.append(make_rows(base, c, "pathwise_max_cqr", info))
            thresholds.append({"origin_date": asof, "family": key[0], "cohort": key[1], "horizon_band": "ALL_28",
                               "method": key[0]+"_pathwise_max_cqr", "correction": c, **info,
                               "origin_blocks": sample.origin_date.nunique(), "availability_max": sample.maturity_date.max(),
                               "coverage_target": "complete_28_day_trajectory"})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(thresholds), pd.DataFrame(availability)


def aci_intervals(ledger, end, config):
    """At day close, process mature events in fixed order BEFORE issuing new intervals."""
    start, end = pd.Timestamp(config["aci_start"]), pd.Timestamp(end)
    issues = set(pd.to_datetime(ledger.loc[ledger.origin_date.between(start, end), "origin_date"].unique()))
    keys = list(ledger[GROUP].drop_duplicates().itertuples(index=False, name=None))
    states, counts = {key: config["alpha"] for key in keys}, {key: 0 for key in keys}
    actuals = ledger.set_index("score_id").actual.to_dict()
    pending = defaultdict(list)
    frames, thresholds, events = [], [], []
    for date in pd.date_range(start, end):
        # Due date is observation availability, which may lag the target date.
        due = sorted(pending.pop(date, []), key=lambda row: (row["origin_date"], row["store_id"], row["item_id"], row["horizon"], row["family"]))
        for event in due:
            key = (event["family"], event["cohort"], event["horizon_band"])
            actual = float(actuals[event["score_id"]])
            if not np.isfinite(actual) or actual < 0 or event["target_date"] > date:
                raise ValueError("Immature or invalid ACI feedback label")
            miss = int(actual < event["lower"] or actual > event["upper"])
            before = states[key]
            after, clipped = update_alpha(before, miss, config["alpha"], config["aci_gamma"],
                                           (config["aci_alpha_min"], config["aci_alpha_max"]))
            states[key], counts[key] = after, counts[key]+1
            events.append({**event, "feedback_date": date, "actual": actual, "miss": miss,
                           "alpha_before": before, "alpha_after": after, "bounds_clipped": clipped,
                           "gamma": config["aci_gamma"], "mature_feedback_count": counts[key]})
        if date not in issues:
            continue
        recent = recent_pool(ledger, date, config["rolling_scores"])
        grouped = dict(tuple(recent.groupby(GROUP, sort=True))) if len(recent) else {}
        forecast = ledger[ledger.origin_date.eq(date)]
        for key, base in forecast.groupby(GROUP, sort=True):
            sample = grouped.get(key, recent.iloc[:0])
            c, info = corrected(sample.cqr_signed.to_numpy() if len(sample) else [], states[key])
            frame = make_rows(base, c, "aci_cqr", info)
            frame["mature_feedback_count"] = counts[key]
            frames.append(frame)
            thresholds.append(dict(zip(GROUP, key), origin_date=date, method=key[0]+"_aci_cqr", correction=c,
                                   mature_feedback_count=counts[key], gamma=config["aci_gamma"], **info, **audit(sample)))
            due_dates = base.set_index("score_id").observation_available_date.to_dict()
            for row in frame.to_dict("records"):
                due_date = pd.Timestamp(due_dates[row["score_id"]])
                if due_date <= date:
                    raise ValueError("ACI cannot observe an outcome at issue time")
                pending[due_date].append({k: row[k] for k in [*KEYS, "family", "cohort", "horizon_band", "score_id", "lower", "upper"]})
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return output, pd.DataFrame(thresholds), pd.DataFrame(events)
