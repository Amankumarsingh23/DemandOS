"""Target-only adapters over the existing validated indexed daily panel."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from demandos.neural.sequences import validate_examples


SCALING = {"method": "own_origin_observed_mean", "floor": 1., "global_statistics": None,
           "padding": "left_zero_with_binary_valid_mask", "chronology": "oldest_to_newest"}


class TargetDataset(Dataset):
    """X/mask [N,56], scale [N]; separate optional Y [N,28]. No covariates."""
    def __init__(self, panel, rows, config, *, label_cutoff=None):
        validate_examples(rows, config, label_cutoff)
        if len(rows) > config.max_windows:
            raise ValueError("Target sequence cache cap exceeded")
        self.rows, self.audits = rows.reset_index(drop=True).copy(), []
        self.label_cutoff = pd.Timestamp(label_cutoff) if label_cutoff is not None else None
        histories, masks, scales, labels = [], [], [], []
        for row in self.rows.itertuples():
            origin = pd.Timestamp(row.origin_date)
            dates = pd.date_range(origin-pd.Timedelta(days=config.lookback-1), origin)
            key = (row.store_id, row.item_id)
            if key not in panel.metadata.index:
                raise ValueError("Unknown source series key")
            meta = panel.metadata.loc[key]
            supported = pd.Timestamp(meta.first_observed_date) if "first_observed_date" in meta and pd.notna(meta.first_observed_date) else None
            y, mask = np.zeros(config.lookback, "float32"), np.zeros(config.lookback, "float32")
            if key in panel.series:
                observed_dates, values = panel.series[key]
                positions = observed_dates.get_indexer(dates)
                if ((positions < 0) & (dates >= (supported if supported is not None else observed_dates[0]))).any():
                    raise ValueError("Missing-source record; only pre-support left padding is permitted")
                valid = positions >= 0
                y[valid], mask[valid] = values[positions[valid]], 1.
            elif supported is not None and (dates >= supported).any():
                raise ValueError("Missing supported history")
            count = int(mask.sum())
            scale = max(1., float(np.abs(y).sum())/max(1, count))
            status = "no_history" if count == 0 else "all_zero" if y.sum() == 0 else "short_history" if count < config.lookback else "complete"
            histories.append(y/scale); masks.append(mask); scales.append(scale)
            self.audits.append({"store_id": row.store_id, "item_id": row.item_id, "origin_date": origin,
                                "history_start": dates[0], "history_end": dates[-1], "observed_days": count,
                                "history_status": status, "scale": scale, "target_end": origin+pd.Timedelta(days=config.horizon)})
            if label_cutoff is not None:
                labels.append(panel.labels(row.store_id, row.item_id, origin, config, label_cutoff)/scale)
        arrays = [np.asarray(histories, "float32"), np.asarray(masks, "float32"), np.asarray(scales, "float32")]
        if label_cutoff is not None:
            arrays.append(np.asarray(labels, "float32"))
        if not all(np.isfinite(x).all() for x in arrays):
            raise ValueError("Nonfinite target-only tensors")
        self.tensors = tuple(torch.from_numpy(x) for x in arrays)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)
