"""One origin row and one complete 28-target vector per vector-output sample."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from demandos.boosting.models import TARGET, validate_rows
from demandos.boosting.dataset import eligible_rows
from demandos.features.contracts import KEYS


ORIGIN_KEYS = ["store_id", "item_id", "origin_date"]
HORIZONS = tuple(range(1, 29))


def complete_rows(pool, cutoff):
    allowed = eligible_rows(pool, cutoff)
    groups = allowed.groupby(ORIGIN_KEYS, sort=True).horizon.agg(["size", "nunique"])
    keys = groups[groups["size"].eq(28) & groups["nunique"].eq(28)].reset_index()[ORIGIN_KEYS]
    result = allowed.merge(keys, on=ORIGIN_KEYS, validate="many_to_one")
    if result.empty or (pd.to_datetime(result.origin_date) + pd.Timedelta(days=28)).gt(pd.Timestamp(cutoff)).any():
        raise ValueError("Multi-output requires complete observed 28-target vectors")
    return result


@dataclass(frozen=True)
class VectorLayout:
    shared: list
    calendar: list

    @classmethod
    def from_long(cls, frame):
        validate_rows(frame)
        calendar = [c for c in frame if c.startswith("target_") and c != "target_date"]
        shared = [c for c in frame if c not in ["horizon", "target_date", *calendar]]
        return cls(shared, calendar)

    @property
    def columns(self):
        return [*self.shared, *(f"{c}__h{h:02}" for h in HORIZONS for c in self.calendar)]

    def pack(self, frame):
        validate_rows(frame)
        if set(frame) != set(self.shared + self.calendar + ["horizon", "target_date"]):
            raise ValueError("Vector input must match inference-only long schema")
        counts = frame.groupby(ORIGIN_KEYS).horizon.size()
        if not counts.eq(28).all():
            raise ValueError("Every inference vector requires horizons 1..28")
        nonkeys = [c for c in self.shared if c not in ORIGIN_KEYS]
        if frame.groupby(ORIGIN_KEYS)[nonkeys].nunique(dropna=False).gt(1).any().any():
            raise ValueError("Origin history must be shared across vector horizons")
        wide = frame[frame.horizon.eq(1)][self.shared].sort_values(ORIGIN_KEYS).reset_index(drop=True)
        parts = [wide]
        for h in HORIZONS:
            selected = frame[frame.horizon.eq(h)].sort_values(ORIGIN_KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(wide[ORIGIN_KEYS], selected[ORIGIN_KEYS])
            parts.append(selected[self.calendar].rename(columns={name: f"{name}__h{h:02}" for name in self.calendar}))
        return pd.concat(parts, axis=1)[self.columns].copy()

    def unpack(self, wide):
        if wide.empty or set(wide) != set(self.columns) or wide.duplicated(ORIGIN_KEYS).any():
            raise ValueError("Unique inference-only origin-vector schema required")
        blocks = []
        for h in HORIZONS:
            block = wide[self.shared].copy()
            for name in self.calendar:
                block[name] = wide[f"{name}__h{h:02}"]
            block["horizon"] = h
            block["target_date"] = pd.to_datetime(block.origin_date) + pd.Timedelta(days=h)
            blocks.append(block)
        result = pd.concat(blocks, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        validate_rows(result)
        return result


def target_vectors(rows):
    keys = rows[ORIGIN_KEYS].drop_duplicates().sort_values(ORIGIN_KEYS).reset_index(drop=True)
    result = keys.copy()
    for h in HORIZONS:
        part = rows[rows.horizon.eq(h)].sort_values(ORIGIN_KEYS).reset_index(drop=True)
        if len(part) != len(keys):
            raise ValueError("Incomplete vector targets; partial masks unsupported")
        pd.testing.assert_frame_equal(keys, part[ORIGIN_KEYS])
        result[f"y_{h:02}"] = part[TARGET]
    return result


def attach_vectors(layout, X, Y, cutoff):
    expected = ORIGIN_KEYS + [f"y_{h:02}" for h in HORIZONS]
    if set(Y) != set(expected) or Y.duplicated(ORIGIN_KEYS).any() or Y[expected].isna().any().any():
        raise ValueError("Unique complete 28-target vectors required")
    values = Y[[f"y_{h:02}" for h in HORIZONS]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Vector targets must be nonnegative observed integer sales")
    if (pd.to_datetime(Y.origin_date) + pd.Timedelta(days=28)).gt(pd.Timestamp(cutoff)).any():
        raise ValueError("Entire target vector must be observed by cutoff")
    if len(X) != len(Y) or len(X[ORIGIN_KEYS].merge(Y[ORIGIN_KEYS], on=ORIGIN_KEYS, validate="one_to_one")) != len(X):
        raise ValueError("Feature/target vector keys must match exactly")
    labelled = []
    for h in HORIZONS:
        part = Y[ORIGIN_KEYS + [f"y_{h:02}"]].rename(columns={f"y_{h:02}": TARGET})
        part["horizon"] = h
        labelled.append(part)
    result = layout.unpack(X).merge(pd.concat(labelled), on=KEYS, validate="one_to_one")
    validate_rows(result, labelled=True, cutoff=cutoff)
    return result
