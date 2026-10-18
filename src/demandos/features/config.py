"""Small, explicit feature definitions and bounded demonstration settings."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple = (1, 7, 14, 28, 56)
    windows: tuple = (7, 14, 28)
    training_origins: tuple = ("2015-10-11", "2015-11-08", "2015-12-06", "2016-01-03")
    mapping_fit_end: str = "2015-10-10"
    max_store_items: int = 14
    max_expanded_series: int = 256
    full_origin_pass: bool = True

    def __post_init__(self):
        for values, maximum in ((self.lags, 365), (self.windows, 365)):
            if not values or len(set(values)) != len(values) or any(type(n) is not int or not 1 <= n <= maximum for n in values):
                raise ValueError("Unique positive integer lags/windows <=365 required")
        if not {1, 7, 14, 28}.issubset(self.lags) or not {7, 14, 28}.issubset(self.windows):
            raise ValueError("Core lag/window definitions must be retained")
        if type(self.max_store_items) is not int or not 1 <= self.max_store_items <= 30:
            raise ValueError("Default cohort must have 1..30 series")
        if type(self.max_expanded_series) is not int or not 1 <= self.max_expanded_series <= 1024:
            raise ValueError("Expanded cohort cap must be in 1..1024")
        if type(self.full_origin_pass) is not bool or not self.training_origins or len(self.training_origins) > 24:
            raise ValueError("Invalid full-pass flag or training-origin count")
        dates = pd.DatetimeIndex(pd.to_datetime(self.training_origins))
        cutoff = pd.Timestamp(self.mapping_fit_end)
        if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or not dates.equals(dates.normalize()):
            raise ValueError("Unique ordered midnight training origins required")
        if pd.isna(cutoff) or cutoff != cutoff.normalize() or cutoff >= dates.min():
            raise ValueError("Mapping/selection fit must precede every training origin")

    @property
    def lookback(self):
        return max(*self.lags, *(w - 1 for w in self.windows), 13)


def load_config(path):
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    if set(data) != {"features"}:
        raise ValueError("Expected [features] configuration")
    return FeatureConfig(**data["features"])
