"""Predeclared, bounded benchmark settings; no evaluation-driven search."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

import pandas as pd


LIBRARIES = ("lightgbm", "xgboost", "catboost")


@dataclass(frozen=True)
class BoostConfig:
    training_start: str = "2014-01-05"
    training_stride: int = 14
    selection_cutoff: str = "2014-01-04"
    validation_origin: str = "2015-12-06"
    training_series: int = 96
    max_rows: int = 200000
    max_origins: int = 64
    iterations: int = 160
    patience: int = 15
    threads: int = 2
    seed: int = 42
    learning_rate: float = .05
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 20
    min_child_weight: float = 5.
    reg_alpha: float = 0.
    reg_lambda: float = 1.
    subsample: float = .9
    colsample: float = .9
    cat_l2: float = 3.
    permutation_rows: int = 1024
    permutation_features: int = 5

    def __post_init__(self):
        bounds = {"training_stride": (1, 28), "training_series": (2, 512), "max_rows": (100, 1000000),
                  "max_origins": (2, 128), "iterations": (1, 500), "patience": (1, 50), "threads": (1, 4),
                  "max_depth": (2, 10), "num_leaves": (2, 128), "min_child_samples": (2, 500),
                  "permutation_rows": (10, 4096), "permutation_features": (1, 10)}
        for name, (lower, upper) in bounds.items():
            if type(getattr(self, name)) is not int or not lower <= getattr(self, name) <= upper:
                raise ValueError(f"Invalid bounded parameter {name}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("Valid deterministic seed required")
        if not 0 < self.learning_rate <= .3 or not 0 < self.subsample <= 1 or not 0 < self.colsample <= 1:
            raise ValueError("Invalid learning/sampling rate")
        if min(self.reg_alpha, self.reg_lambda, self.min_child_weight, self.cat_l2) < 0:
            raise ValueError("Regularization cannot be negative")
        start, selection, validation = [pd.Timestamp(getattr(self, key)) for key in ("training_start", "selection_cutoff", "validation_origin")]
        if any(pd.isna(t) or t != t.normalize() for t in (start, selection, validation)) or not selection < start < validation:
            raise ValueError("Selection must precede training, which precedes validation")


def load_config(path):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"boosting"}:
        raise ValueError("Expected [boosting] configuration")
    return BoostConfig(**raw["boosting"])
