"""Indexed daily windows; predictors and labels are deliberately separate APIs."""

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


STATIC = ("store_id", "state_id", "cat_id", "dept_id", "item_id")
CALENDAR = ("weekday_sin", "weekday_cos", "month_sin", "month_cos", "event", "snap")
PAST = ("scaled_sales", "observed", "price_unavailable", "price_observed", *CALENDAR)
FUTURE = (*CALENDAR, "horizon_fraction")
KEY = ["store_id", "item_id", "origin_date"]


def daily_dates(values):
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.empty or dates.hasnans or dates.has_duplicates or not dates.equals(dates.normalize()):
        raise ValueError("Unique daily midnight dates required")
    if not dates.equals(pd.date_range(dates.min(), dates.max(), freq="D")):
        raise ValueError("Missing-source daily gap or unordered dates")
    return dates


class Panel:
    """Small warehouse projection indexed once, independent of future price/label columns."""
    def __init__(self, daily, calendar, metadata):
        if metadata.empty or metadata[list(STATIC)].isna().any().any() or metadata.duplicated(["store_id", "item_id"]).any():
            raise ValueError("Unique complete static metadata required")
        self.metadata = metadata.set_index(["store_id", "item_id"], drop=False).copy()
        self.calendar = calendar.copy()
        self.calendar_dates = daily_dates(calendar.date)
        self.cal = {}
        for state in ("CA", "TX", "WI"):
            snap = calendar[f"snap_{state}"].to_numpy(float)
            if not np.isfinite(snap).all() or not np.isin(snap, [0, 1]).all():
                raise ValueError("Invalid state SNAP calendar")
            dow, month = self.calendar_dates.dayofweek, self.calendar_dates.month - 1
            self.cal[state] = np.column_stack([
                np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
                np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
                (calendar.event_name_1.notna() | calendar.event_name_2.notna()).astype(float), snap]).astype("float32")
        self.series = {}
        for key, group in daily.groupby(["store_id", "item_id"], sort=True):
            if key not in self.metadata.index:
                raise ValueError("Unexpected source series")
            dates = daily_dates(group.date)
            y = group.units_sold.to_numpy(float)
            if not np.isfinite(y).all() or (y < 0).any() or (y != np.floor(y)).any():
                raise ValueError("Invalid observed sales")
            self.series[key] = (dates, y.astype("float32"))
        if not self.metadata.state_id.isin(self.cal).all():
            raise ValueError("Unsupported state")

    def inputs(self, store, item, origin, config):
        origin = pd.Timestamp(origin)
        if pd.isna(origin) or origin != origin.normalize():
            raise ValueError("Daily origin required")
        key = (store, item)
        if key not in self.metadata.index:
            raise ValueError("Unknown static source key")
        meta = self.metadata.loc[key]
        past_dates = pd.date_range(origin-pd.Timedelta(days=config.lookback-1), origin)
        future_dates = pd.date_range(origin+pd.Timedelta(days=1), periods=config.horizon)
        cp = self.calendar_dates.get_indexer(past_dates)
        cf = self.calendar_dates.get_indexer(future_dates)
        if (cp < 0).any() or (cf < 0).any():
            raise ValueError("Calendar does not cover exact sequence dates")
        values, mask = np.zeros(config.lookback, "float32"), np.zeros(config.lookback, "float32")
        supported_start = pd.Timestamp(meta["first_observed_date"]) if "first_observed_date" in meta and pd.notna(meta["first_observed_date"]) else None
        if key in self.series:
            dates, y = self.series[key]
            positions = dates.get_indexer(past_dates)
            # Padding is allowed only before the first source record. Tail/internal gaps fail.
            missing_source = (positions < 0) & (past_dates >= (supported_start if supported_start is not None else dates[0]))
            if missing_source.any():
                raise ValueError("Missing-source history records, not short-history padding")
            observed = positions >= 0
            values[observed], mask[observed] = y[positions[observed]], 1
        elif supported_start is not None and (past_dates >= supported_start).any():
            raise ValueError("Missing-source history records for a known supported series")
        count = int(mask.sum())
        scale = max(1., float(values.sum()) / max(1, count))
        status = "no_history" if count == 0 else "all_zero" if values.sum() == 0 else "short_history" if count < config.lookback else "complete"
        # Price availability is explicitly unknown, with a consumed zero availability mask.
        past = np.column_stack([values/scale, mask, np.zeros((config.lookback, 2)), self.cal[meta.state_id][cp]]).astype("float32")
        future = np.column_stack([self.cal[meta.state_id][cf], np.arange(1, config.horizon+1)/config.horizon]).astype("float32")
        audit = {"history_start": past_dates[0], "history_end": past_dates[-1], "target_start": future_dates[0],
                 "target_end": future_dates[-1], "history_observed": count, "history_status": status,
                 "short_history": count < config.lookback, "scale": scale}
        return past, future, meta, audit

    def labels(self, store, item, origin, config, cutoff):
        origin, cutoff = pd.Timestamp(origin), pd.Timestamp(cutoff)
        if pd.isna(origin) or pd.isna(cutoff) or origin != origin.normalize() or cutoff != cutoff.normalize():
            raise ValueError("Explicit daily label origin and cutoff required")
        dates = pd.date_range(origin+pd.Timedelta(days=1), periods=config.horizon)
        if dates[-1] > cutoff:
            raise ValueError("Full target vector exceeds allowed fitting cutoff")
        key = (store, item)
        if key not in self.series:
            raise ValueError("Missing training labels")
        observed, values = self.series[key]
        index = observed.get_indexer(dates)
        if (index < 0).any():
            raise ValueError("Missing full-vector training labels")
        return values[index].copy()


def examples(keys, origins, config, cutoff=None):
    if keys.empty or keys[["store_id", "item_id"]].isna().any().any() or keys.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Unique nonempty example identities required")
    dates = pd.DatetimeIndex(origins)
    if dates.has_duplicates or dates.hasnans or not dates.equals(dates.normalize()) or not dates.is_monotonic_increasing:
        raise ValueError("Unique chronological origins required")
    if cutoff is not None:
        dates = dates[dates+pd.Timedelta(days=config.horizon) <= pd.Timestamp(cutoff)]
    count = len(keys)*len(dates)
    if count == 0 or count > config.max_windows:
        raise ValueError("Empty eligible sequence set or configured window cap exceeded")
    return pd.DataFrame([{"store_id": r.store_id, "item_id": r.item_id, "origin_date": origin}
                         for origin in dates for r in keys.sort_values(["store_id", "item_id"]).itertuples()])


@dataclass
class Preprocessor:
    mean: float
    std: float
    vocabulary: dict
    fit_cutoff: str
    fit_windows: int
    source_max: str

    @classmethod
    def fit(cls, panel, rows, config, cutoff):
        validate_examples(rows, config, cutoff)
        sums, squares, count, vocabulary = 0., 0., 0, {name: set() for name in STATIC}
        for row in rows.itertuples():
            past, _, meta, _ = panel.inputs(row.store_id, row.item_id, row.origin_date, config)
            # Verify target availability as well as the declared date cutoff.
            panel.labels(row.store_id, row.item_id, row.origin_date, config, cutoff)
            x = past[past[:, 1] == 1, 0].astype(float)
            sums += x.sum(); squares += np.square(x).sum(); count += len(x)
            for name in STATIC:
                vocabulary[name].add(str(meta[name]))
        mean = sums / max(count, 1)
        std = max(float(np.sqrt(max(0., squares/max(count, 1)-mean**2))), 1e-6)
        vocab = {name: {value: i+1 for i, value in enumerate(sorted(values))} for name, values in vocabulary.items()}
        return cls(float(mean), std, vocab, str(pd.Timestamp(cutoff).date()), len(rows), str(rows.origin_date.max().date()))

    def transform(self, past, meta):
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError("Invalid persisted normalization state")
        result = past.copy()
        result[:, 0] = (result[:, 0]-self.mean)/self.std * result[:, 1]
        ids = np.array([self.vocabulary[name].get(str(meta[name]), 0) for name in STATIC], dtype="int64")
        return result, ids

    def state(self):
        return asdict(self)


def validate_examples(rows, config, cutoff=None):
    if rows.empty or list(rows.columns) != KEY or rows.isna().any().any() or rows.duplicated(KEY).any():
        raise ValueError("Exact unique inference keys required, without labels")
    dates = pd.DatetimeIndex(pd.to_datetime(rows.origin_date))
    if not dates.equals(dates.normalize()):
        raise ValueError("Daily origins required")
    if cutoff is not None:
        allowed = pd.Timestamp(cutoff)
        if pd.isna(allowed) or allowed != allowed.normalize() or (dates+pd.Timedelta(days=config.horizon) > allowed).any():
            raise ValueError("Partial future target vectors forbidden for training")


class SequenceDataset(Dataset):
    """Bounded tensor cache of indexed windows, not an expansion of the M5 fact table."""
    def __init__(self, panel, rows, processor, config, *, label_cutoff=None):
        validate_examples(rows, config, label_cutoff)
        if len(rows) > config.max_windows:
            raise ValueError("Sequence memory cap exceeded")
        if pd.to_datetime(rows.origin_date).min() < pd.Timestamp(processor.fit_cutoff) and label_cutoff is None:
            raise ValueError("Preprocessor cannot predict before its fitting cutoff")
        self.rows, self.audits = rows.reset_index(drop=True).copy(), []
        pasts, futures, categories, scales, labels = [], [], [], [], []
        for row in self.rows.itertuples():
            past, future, meta, audit = panel.inputs(row.store_id, row.item_id, row.origin_date, config)
            past, ids = processor.transform(past, meta)
            pasts.append(past); futures.append(future); categories.append(ids); scales.append(audit["scale"])
            self.audits.append({**{key: getattr(row, key) for key in KEY}, **audit})
            if label_cutoff is not None:
                labels.append(panel.labels(row.store_id, row.item_id, row.origin_date, config, label_cutoff)/audit["scale"])
        arrays = [np.asarray(pasts, "float32"), np.asarray(futures, "float32"), np.asarray(categories, "int64"), np.asarray(scales, "float32")]
        if labels:
            arrays.append(np.asarray(labels, "float32"))
        if any(not np.isfinite(x).all() for x in arrays):
            raise ValueError("Nonfinite tensor data")
        self.tensors = tuple(torch.from_numpy(x) for x in arrays)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)
