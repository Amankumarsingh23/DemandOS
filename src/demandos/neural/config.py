"""Predeclared neural budget; evaluation outcomes never select these settings."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class NeuralConfig:
    lookback: int = 56
    horizon: int = 28
    hidden_size: int = 32
    embedding_dim: int = 4
    head_size: int = 32
    dropout: float = .1
    batch_size: int = 128
    epochs: int = 12
    patience: int = 3
    learning_rate: float = .001
    weight_decay: float = .0001
    clip_norm: float = 1.
    threads: int = 2
    seed: int = 42
    device: str = "cpu"
    max_windows: int = 8192

    def __post_init__(self):
        bounds = {"lookback": (56, 56), "horizon": (28, 28), "hidden_size": (8, 64),
                  "embedding_dim": (1, 8), "head_size": (8, 64), "batch_size": (1, 512),
                  "epochs": (1, 30), "patience": (1, 10), "threads": (1, 4),
                  "max_windows": (1, 32768), "seed": (0, 2**31-1)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid bounded {name}")
        if not 0 <= self.dropout < 1 or not 0 < self.learning_rate <= .01:
            raise ValueError("Invalid dropout or learning rate")
        if not math.isfinite(self.weight_decay) or not 0 <= self.weight_decay <= 1:
            raise ValueError("Invalid weight decay")
        if not math.isfinite(self.clip_norm) or not 0 < self.clip_norm <= 100:
            raise ValueError("Positive finite gradient clip required")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Only explicit cpu/cuda devices supported; mixed precision disabled")


def load_config(path):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"neural"}:
        raise ValueError("Expected [neural] settings")
    return NeuralConfig(**raw["neural"])
