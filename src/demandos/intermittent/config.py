"""Bounded, predeclared Season 11 settings."""
from dataclasses import dataclass
from pathlib import Path
import math
import tomllib


@dataclass(frozen=True)
class Config:
    boosting_config: str = "configs/boosting.toml"
    history_window: int = 365
    trailing_window: int = 56
    adi_boundary: float = 1.32
    cv2_boundary: float = .49
    short_history: int = 28
    prior_window: int = 56
    kappa: float = 28.
    minimum_prior_rows: int = 56
    minimum_validation_series: int = 5
    minimum_validation_rows: int = 140
    stress_per_group: int = 4
    cold_per_mode: int = 3
    seed: int = 42

    def __post_init__(self):
        for name, lower, upper in (("history_window",28,1913),("trailing_window",14,365),("short_history",7,28),
            ("prior_window",7,365),("minimum_prior_rows",1,10000),("minimum_validation_series",2,96),
            ("minimum_validation_rows",28,2688),("stress_per_group",1,8),("cold_per_mode",1,4)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"Invalid bounded {name}")
        if any(not math.isfinite(v) or v <= 0 for v in (self.adi_boundary,self.cv2_boundary,self.kappa)):
            raise ValueError("Finite positive thresholds/shrinkage strength required")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("Invalid deterministic seed")
        path = Path(self.boosting_config)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Boosting configuration must be project-relative")


def load_config(path):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"intermittent"}:
        raise ValueError("Expected [intermittent] config")
    return Config(**raw["intermittent"])
