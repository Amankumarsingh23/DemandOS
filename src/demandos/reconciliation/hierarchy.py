"""Canonical crossed groupings: every bottom belongs to one node per level."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy import sparse


LEVELS = {
    "TOTAL": (), "STATE": ("state_id",), "STORE": ("store_id",),
    "CATEGORY": ("cat_id",), "DEPARTMENT": ("dept_id",),
    "STATE_CATEGORY": ("state_id", "cat_id"), "STATE_DEPARTMENT": ("state_id", "dept_id"),
    "STORE_CATEGORY": ("store_id", "cat_id"), "STORE_DEPARTMENT": ("store_id", "dept_id"),
    "ITEM": ("item_id",), "ITEM_STATE": ("item_id", "state_id"),
    "ITEM_STORE": ("store_id", "item_id"),
}
KEYS = ["store_id", "item_id"]
EXPECTED_COUNTS = [1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490]


@dataclass
class Hierarchy:
    bottom: pd.DataFrame
    nodes: pd.DataFrame
    membership: pd.DataFrame
    S: sparse.csr_matrix

    def rows(self, level):
        return self.nodes.index[self.nodes.level.eq(level)].to_numpy()

    @property
    def bottom_rows(self):
        return self.rows("ITEM_STORE")

    def check(self, *, full=False):
        n = len(self.bottom)
        if not sparse.isspmatrix_csr(self.S) or self.S.shape != (len(self.nodes), n):
            raise ValueError("Invalid sparse summing matrix")
        if self.nodes.node_id.duplicated().any() or not np.all(self.S.data == 1):
            raise ValueError("Duplicate node or invalid membership")
        if (self.S[self.bottom_rows] != sparse.eye(n, format="csr")).nnz:
            raise ValueError("Bottom identity block is misaligned")
        for level in LEVELS:
            if not np.all(np.asarray(self.S[self.rows(level)].sum(axis=0)).ravel() == 1):
                raise ValueError("Every bottom must have exactly one membership per level")
        if self.S.nnz != 12*n or np.any(np.diff(self.S.indptr) == 0):
            raise ValueError("Empty or duplicate membership")
        if full and (n != 30490 or [len(self.rows(k)) for k in LEVELS] != EXPECTED_COUNTS):
            raise ValueError("Full M5 hierarchy counts differ from official groupings")
        return {"shape": list(self.S.shape), "nnz": self.S.nnz,
                "sparse_bytes": self.S.data.nbytes+self.S.indices.nbytes+self.S.indptr.nbytes,
                "level_counts": {k: len(self.rows(k)) for k in LEVELS}}


def build(metadata, *, scope="COHORT"):
    columns = ["store_id", "item_id", "state_id", "cat_id", "dept_id"]
    if metadata.empty or metadata[columns].isna().any().any() or metadata.duplicated(KEYS).any():
        raise ValueError("Unique nonempty complete bottom dimensions required")
    bottom = metadata[columns].sort_values(KEYS).reset_index(drop=True).copy()
    for key, values in [("store_id", ["state_id"]), ("item_id", ["cat_id", "dept_id"]), ("dept_id", ["cat_id"])]:
        if bottom.groupby(key)[values].nunique().gt(1).any().any():
            raise ValueError("Inconsistent store/product dimension relationships")
    nodes, entries = [], []
    for level, dimensions in LEVELS.items():
        groups = sorted(set(bottom[list(dimensions)].itertuples(index=False, name=None))) if dimensions else [()]
        lookup = {}
        for values in groups:
            row = len(nodes)
            lookup[values] = row
            nodes.append({"node_id": f"{scope}_{level}:"+json.dumps(values, separators=(",", ":")),
                          "level": level, "scope": scope, **dict(zip(dimensions, values))})
        for col, record in enumerate(bottom.to_dict("records")):
            row = lookup[tuple(record[k] for k in dimensions)]
            entries.append((row, col, level))
    membership = pd.DataFrame(entries, columns=["node_row", "bottom_col", "level"])
    S = sparse.csr_matrix((np.ones(len(entries), dtype="float64"),
                           (membership.node_row, membership.bottom_col)), shape=(len(nodes), len(bottom)))
    bottom["bottom_col"] = np.arange(len(bottom))
    nodes = pd.DataFrame(nodes).rename_axis("node_row").reset_index()
    nodes["members"] = np.diff(S.indptr)
    result = Hierarchy(bottom, nodes, membership, S)
    result.check()
    return result


def dense_cohort(S, maximum=256):
    """Guard BEFORE conversion; full M5 must never enter a dense solver."""
    if not sparse.issparse(S) or S.shape[1] > maximum or S.shape[0] > 12*maximum:
        raise ValueError("Dense reconciliation is restricted to a bounded cohort")
    return S.toarray()


def observed_matrix(daily, hierarchy):
    """No padding missing source records with zeros."""
    if daily.duplicated(KEYS+["date"]).any():
        raise ValueError("Duplicate observed daily keys")
    pivot = daily.pivot(index=KEYS, columns="date", values="units_sold")
    index = pd.MultiIndex.from_frame(hierarchy.bottom[KEYS])
    pivot = pivot.reindex(index)
    dates = pd.DatetimeIndex(pivot.columns)
    values = pivot.to_numpy(float)
    if (dates.empty or dates.hasnans or not dates.equals(dates.normalize())
            or not dates.equals(pd.date_range(dates.min(), dates.max())) or not np.isfinite(values).all()
            or (values < 0).any() or (values != np.floor(values)).any()):
        raise ValueError("Complete aligned daily nonnegative integer observations required")
    if set(pivot.index) != set(pd.MultiIndex.from_frame(daily[KEYS].drop_duplicates())):
        raise ValueError("Unexpected observed series")
    return dates, values
