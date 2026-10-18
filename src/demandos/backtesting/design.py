"""Validated daily schedules and predeclared, bounded experiment configuration."""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

import pandas as pd

from demandos.forecasting.models import MODELS
from demandos.forecasting.pipeline import ForecastConfig
from demandos.timeseries.temporal import origin_schedule


@dataclass(frozen=True)
class BacktestConfig:
    first_origin: str = "2016-01-31"
    last_origin: str = "2016-03-27"
    horizon: int = 28
    stride: int = 28
    minimum_train_days: int = 84
    sliding_days: int = 730
    windows: tuple = ("expanding", "sliding")
    models: tuple = MODELS
    max_store_items: int = 14
    store_aggregates: int = 1
    category_aggregates: int = 3
    arima_series: int = 8
    sarima_series: int = 4
    model_config: ForecastConfig = field(default_factory=ForecastConfig)

    def __post_init__(self):
        for name in ("horizon", "stride", "minimum_train_days", "sliding_days", "max_store_items",
                     "store_aggregates", "category_aggregates", "arima_series", "sarima_series"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Positive integer required: {name}")
        if self.horizon > 28 or self.sliding_days < self.minimum_train_days or self.sliding_days > 3650:
            raise ValueError("Horizon/window exceeds bounded experiment limits")
        if self.max_store_items > 30 or self.store_aggregates > 10 or self.category_aggregates > 3:
            raise ValueError("Cohort exceeds small-cohort limits")
        if self.arima_series > 16 or self.sarima_series > 8:
            raise ValueError("ARIMA/SARIMA cohort exceeds bounded limits")
        if not self.models or len(set(self.models)) != len(self.models) or set(self.models) - set(MODELS):
            raise ValueError("Unique supported models required")
        if not self.windows or len(set(self.windows)) != len(self.windows) or set(self.windows) - {"expanding", "sliding"}:
            raise ValueError("Unique expanding/sliding windows required")
        first, last = pd.Timestamp(self.first_origin), pd.Timestamp(self.last_origin)
        if pd.isna(first) or pd.isna(last) or first != first.normalize() or last != last.normalize() or first > last:
            raise ValueError("Invalid origin bounds")
        if (last - first).days // self.stride + 1 > 12:
            raise ValueError("At most 12 origins in this bounded experiment")


def schedules(dates, config):
    index = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    first, last = pd.Timestamp(config.first_origin), pd.Timestamp(config.last_origin)
    if index.empty or index.hasnans or first not in index or last not in index or last + pd.Timedelta(days=config.horizon) > index[-1]:
        raise ValueError("Origin or complete targets outside observed calendar")
    initial = int((first - index[0]).days) + 1
    if initial < config.minimum_train_days:
        raise ValueError("Insufficient initial training history")
    frames = []
    for window in config.windows:
        size = None if window == "expanding" else config.sliding_days
        if size is not None and initial < size:
            raise ValueError("Sliding window must be fully observed at first origin")
        frame = origin_schedule(index, initial, config.horizon, config.stride, size)
        frame = frame[frame.origin.le(last)].copy()
        frame["window"] = window
        frame["training_days"] = (frame.origin - frame.history_start).dt.days + 1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_config(path):
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    if set(data) - {"backtesting", "model"} or "backtesting" not in data:
        raise ValueError("Expected [backtesting] and optional [model]")
    return BacktestConfig(**data["backtesting"], model_config=ForecastConfig(**data.get("model", {})))
