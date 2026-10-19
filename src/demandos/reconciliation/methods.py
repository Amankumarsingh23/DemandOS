"""Stable bounded reconciliation; every published solution is S @ bottom."""

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular, lstsq
from scipy.optimize import nnls

from .hierarchy import dense_cohort


def shares(history, partition, *, supported=None):
    """Prefix volume shares within disjoint groups; no imputed missing histories."""
    history, partition = np.asarray(history, float), np.asarray(partition, float)
    if history.ndim != 2 or history.shape[1] == 0 or not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("Finite nonnegative observed prefix required")
    if partition.ndim != 2 or partition.shape[1] != len(history) or not np.isin(partition, [0, 1]).all():
        raise ValueError("Binary grouping membership required")
    if not np.all(partition.sum(axis=0) == 1) or (partition.sum(axis=1) == 0).any():
        raise ValueError("Middle level must be a nonempty disjoint partition")
    supported = np.ones(len(history), bool) if supported is None else np.asarray(supported, bool)
    if supported.shape != (len(history),):
        raise ValueError("History support must align to bottom nodes")
    allocation = np.zeros_like(partition.T)
    records = []
    volume = history.sum(axis=1)
    for group, members in enumerate(partition):
        eligible = members.astype(bool) & supported
        if not eligible.any():
            raise ValueError("Unavailable allocation: no supported descendants; skip experiment")
        total = volume[eligible].sum()
        allocation[eligible, group] = volume[eligible]/total if total > 0 else 1/eligible.sum()
        records.append({"group": group, "status": "volume" if total > 0 else "equal_zero_volume",
                        "volume": total, "supported_descendants": int(eligible.sum()),
                        "unsupported_descendants": int((members.astype(bool) & ~supported).sum())})
    np.testing.assert_allclose(allocation.sum(axis=0), 1, atol=1e-12)
    return allocation, pd.DataFrame(records)


def projection(S, covariance=None, *, nonnegative=False, maximum=256):
    dense = dense_cohort(S, maximum)
    n = len(dense)
    if covariance is None:
        A, factor = dense, None
    else:
        W = np.asarray(covariance, float)
        if W.shape == (n,):
            if not np.isfinite(W).all() or (W <= 0).any():
                raise ValueError("Strictly positive finite variances required")
            factor = np.sqrt(W)
            A = dense/factor[:, None]
        elif W.shape == (n, n):
            if not np.isfinite(W).all() or not np.allclose(W, W.T):
                raise ValueError("Finite symmetric covariance required")
            factor = cholesky(W, lower=True)
            A = solve_triangular(factor, dense, lower=True)
        else:
            raise ValueError("Covariance node order/shape mismatch")
    # QR solves the weighted problem, without forming an inverse or normal equations.
    diagnostics = {"design_condition": float(np.linalg.cond(A)), "nonnegative": nonnegative}
    if not np.isfinite(diagnostics["design_condition"]) or diagnostics["design_condition"] > 1e12:
        raise ValueError("Ill-conditioned reconciliation design")

    def solve(base):
        start = perf_counter()
        y = np.asarray(base, float)
        if y.ndim != 2 or y.shape[0] != n or not np.isfinite(y).all():
            raise ValueError("Finite aligned all-node forecast matrix required")
        target = y if factor is None else y/factor[:, None] if factor.ndim == 1 else solve_triangular(factor, y, lower=True)
        if nonnegative:
            bottom = np.column_stack([nnls(A, col, maxiter=10*A.shape[1])[0] for col in target.T])
        else:
            bottom, _, rank, _ = lstsq(A, target, lapack_driver="gelsy")
            if rank != S.shape[1]:
                raise ValueError("Rank-deficient reconciliation solve")
        result = S@bottom
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite reconciliation output")
        return result, {**diagnostics, "seconds": perf_counter()-start}
    return solve


def reconcile(hierarchy, base, history, covariance, *, maximum=256):
    S = hierarchy.S
    top, top_status = shares(history, dense_cohort(S[hierarchy.rows("TOTAL")], maximum))
    middle_rows = hierarchy.rows("STORE_DEPARTMENT")
    middle, middle_status = shares(history, dense_cohort(S[middle_rows], maximum))
    outputs = {"bottom_up": S@base[hierarchy.bottom_rows],
               "top_down": S@(top@base[hierarchy.rows("TOTAL")]),
               "middle_out": S@(middle@base[middle_rows])}
    diagnostics = []
    choices = [("ols", None, False),
            ("wls_structural", np.asarray(S.sum(axis=1)).ravel(), False)]
    if covariance["method"] != "structural_fallback":
        choices.append(("wls_variance", covariance["diagonal"], False))
    if covariance["method"] == "mint_shrinkage":
        choices.append(("mint_shrinkage", covariance["W"], False))
    actual_weighting = ("mint_shrinkage" if covariance["method"] == "mint_shrinkage"
                        else "wls_structural" if covariance["method"] == "structural_fallback" else "wls_variance")
    choices.append(("nonnegative_"+actual_weighting, covariance["W"], True))
    for name, W, nonnegative in choices:
        value, info = projection(S, W, nonnegative=nonnegative, maximum=maximum)(base)
        outputs[name] = value
        diagnostics.append({"method": name, **info})
    allocation = {"top": top, "middle": middle, "top_status": top_status, "middle_status": middle_status}
    for name, y in outputs.items():
        gap = float(np.max(np.abs(y-S@y[hierarchy.bottom_rows])))
        if gap > 1e-8:
            raise ValueError(f"Incoherent {name}: {gap}")
    return outputs, diagnostics, allocation
