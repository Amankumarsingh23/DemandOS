"""Validated, portable EDA choices."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import math
import tomllib


@dataclass(frozen=True)
class EDAConfig:
    start_date: str = ""
    end_date: str = ""
    abc_a: float = 0.8
    abc_b: float = 0.95
    xyz_x: float = 0.5
    xyz_y: float = 1.0
    xyz_min_weeks: int = 8
    xyz_min_positive_weeks: int = 2
    xyz_min_mean: float = 1.0
    adi_boundary: float = 1.32
    positive_cv2_boundary: float = 0.49
    min_positive_days: int = 2
    moving_average_days: int = 28
    iqr_multiplier: float = 1.5
    robust_same_weekday_weeks: int = 8
    robust_min_history: int = 4
    robust_z: float = 4.5
    selected_series: int = 6

    def __post_init__(self):
        for name in ("abc_a", "abc_b", "xyz_x", "xyz_y", "xyz_min_mean", "adi_boundary",
                     "positive_cv2_boundary", "iqr_multiplier", "robust_z"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.abc_a < self.abc_b < 1:
            raise ValueError("ABC requires 0 < A < B < 1")
        if not 0 <= self.xyz_x < self.xyz_y or self.xyz_min_mean < 0:
            raise ValueError("Invalid XYZ thresholds")
        for name in ("xyz_min_weeks", "xyz_min_positive_weeks", "min_positive_days",
                     "moving_average_days", "robust_same_weekday_weeks", "robust_min_history", "selected_series"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if self.selected_series > 12 or self.robust_min_history > self.robust_same_weekday_weeks:
            raise ValueError("Invalid selection size or robust history")
        if min(self.adi_boundary, self.positive_cv2_boundary, self.iqr_multiplier, self.robust_z) <= 0:
            raise ValueError("Classification and anomaly thresholds must be positive")
        for value in (self.start_date, self.end_date):
            if value:
                date.fromisoformat(value)
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("Analysis start is after end")


def load_config(path):
    with Path(path).open("rb") as handle:
        content = tomllib.load(handle)
    if set(content) != {"eda"}:
        raise ValueError("Expected one [eda] configuration section")
    return EDAConfig(**content["eda"])
