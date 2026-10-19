"""Frozen, bounded N-BEATS architecture and optimization settings."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

from demandos.neural.config import NeuralConfig


@dataclass(frozen=True)
class NBeatsConfig:
    lookback: int = 56
    horizon: int = 28
    hidden_size: int = 64
    layers: int = 3
    generic_blocks: int = 3
    generic_theta: int = 8
    generic_share: bool = False
    trend_blocks: int = 2
    seasonality_blocks: int = 2
    interpretable_share: bool = True
    trend_degree: int = 2
    harmonics: int = 3
    period: int = 7
    dropout: float = 0.
    batch_size: int = 128
    epochs: int = 15
    patience: int = 3
    learning_rate: float = .001
    weight_decay: float = .0001
    clip_norm: float = 1.
    threads: int = 2
    seed: int = 42
    device: str = "cpu"
    max_windows: int = 8192

    def __post_init__(self):
        NeuralConfig(**{name: getattr(self, name) for name in ("lookback", "horizon", "hidden_size", "dropout", "batch_size", "epochs", "patience", "learning_rate", "weight_decay", "clip_norm", "threads", "seed", "device", "max_windows")})
        for name, low, high in [("layers", 1, 4), ("generic_blocks", 2, 6), ("generic_theta", 1, 32),
                                ("trend_blocks", 1, 4), ("seasonality_blocks", 1, 4), ("trend_degree", 0, 3),
                                ("harmonics", 1, 3), ("period", 7, 7)]:
            if type(getattr(self, name)) is not int or not low <= getattr(self, name) <= high:
                raise ValueError(f"Invalid bounded {name}")
        if type(self.generic_share) is not bool or type(self.interpretable_share) is not bool:
            raise ValueError("Sharing must be explicit booleans")


def load_config(path):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"nbeats"}:
        raise ValueError("Expected [nbeats] configuration")
    return NBeatsConfig(**raw["nbeats"])
