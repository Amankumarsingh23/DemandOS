"""Paired genuine OOS errors, frozen before evaluation; explicit fallback labels."""

import numpy as np
import pandas as pd


def eligible(residuals, nodes, cutoff):
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff):
        raise ValueError("Explicit residual label cutoff required")
    required = ["node_id", "origin_date", "target_date", "horizon", "error", "fit_cutoff", "fit_target_max"]
    if residuals[required].isna().any().any():
        raise ValueError("Complete residual provenance required")
    frame = residuals.copy()
    for col in ["origin_date", "target_date", "fit_cutoff", "fit_target_max"]:
        frame[col] = pd.to_datetime(frame[col])
    if (frame.fit_cutoff.gt(frame.origin_date).any() or frame.fit_target_max.gt(frame.fit_cutoff).any()
            or frame.target_date.ne(frame.origin_date+pd.to_timedelta(frame.horizon, unit="D")).any()
            or not frame.horizon.isin(range(1, 29)).all() or not np.isfinite(frame.error).all()):
        raise ValueError("Invalid OOS error provenance/date/horizon")
    frame = frame[frame.target_date.le(cutoff)].copy()
    key = ["origin_date", "target_date", "horizon"]
    if frame.duplicated(key+["node_id"]).any() or not set(frame.node_id).issubset(set(nodes)):
        raise ValueError("Duplicate/unexpected residual node alignment")
    matrix = frame.pivot(index=key, columns="node_id", values="error").reindex(columns=list(nodes))
    if matrix.isna().any().any():
        raise ValueError("Unpaired residual observations; no silent complete-case deletion")
    return frame, matrix


def estimate(residuals, nodes, cutoff, structural, *, shrinkage=.9, ridge=1e-6,
             min_origins=8, min_rows=56, maximum_condition=1e10):
    if not 0 < shrinkage <= 1 or ridge <= 0:
        raise ValueError("Positive fixed shrinkage/ridge required")
    frame, errors = eligible(residuals, nodes, cutoff)
    n, m = errors.shape
    origins = errors.index.get_level_values("origin_date").unique()
    structural = np.asarray(structural, float)
    if structural.shape != (len(nodes),) or not np.isfinite(structural).all() or (structural <= 0).any():
        raise ValueError("Positive aligned structural variances required")
    diagnostics = {"fit_cutoff": str(pd.Timestamp(cutoff).date()), "paired_rows": n,
                   "origin_blocks": len(origins), "nodes": m, "shrinkage": shrinkage, "ridge": ridge,
                   "target_max": str(frame.target_date.max()),
                   "dependence": "28 correlated horizons per nonoverlapping origin block, not independent replicates",
                   "sample_covariance_full_rank_claim": False}
    if n < 2:
        return {"W": structural, "diagonal": structural, "method": "structural_fallback",
                "diagnostics": {**diagnostics, "fallback": "fewer than two paired errors"}}
    sample = np.cov(errors.to_numpy(float), rowvar=False, ddof=1)
    diagonal = np.diag(sample).copy()
    positive = diagonal[diagonal > 0]
    floor = max(1e-8, (float(np.median(positive)) if len(positive) else 1.)*1e-6)
    diagonal = np.maximum(diagonal, floor)
    # A regularized covariance approximation; never describe it as full-rank empirical covariance.
    W = (1-shrinkage)*sample+shrinkage*np.diag(diagonal)
    W += ridge*float(diagonal.mean())*np.eye(m)
    eigen = np.linalg.eigvalsh(W)
    condition = float(eigen[-1]/eigen[0]) if eigen[0] > 0 else float("inf")
    diagnostics.update(sample_rank=int(np.linalg.matrix_rank(sample)), sample_dimension=m,
                       variance_floor=floor, floored_nodes=int((np.diag(sample) < floor).sum()),
                       regularized_condition=condition, regularized_min_eigenvalue=float(eigen[0]),
                       nominal_rows_exceed_nodes=n > m)
    reason = ("insufficient historical origin blocks/paired rows" if len(origins) < min_origins or n < min_rows
              else "underdetermined empirical covariance (paired rows/rank insufficient for all nodes)"
              if n <= m or diagnostics["sample_rank"] < m
              else "unstable regularized covariance" if not np.isfinite(condition) or condition > maximum_condition else None)
    return {"W": diagonal if reason else W, "diagonal": diagonal,
            "method": "variance_fallback" if reason else "mint_shrinkage",
            "diagnostics": {**diagnostics, "fallback": reason}}
