"""Fixed CPU-first TFT budget, selected before retrospective evaluation."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

from demandos.neural.config import NeuralConfig


@dataclass(frozen=True)
class TFTConfig(NeuralConfig):
    heads: int = 4
    epochs: int = 8
    patience: int = 2

    def __post_init__(self):
        super().__post_init__()
        if type(self.heads) is not int or not 1 <= self.heads <= 8 or self.hidden_size % self.heads:
            raise ValueError("Attention heads must divide the bounded hidden size")


def load_config(path):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"tft"}:
        raise ValueError("Expected [tft] configuration")
    return TFTConfig(**raw["tft"])
