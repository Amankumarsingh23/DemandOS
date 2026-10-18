"""Small preregistered comparison; no test-period model selection."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class StrategyConfig:
    boosting_config: str = "configs/boosting.toml"
    max_native_models: int = 232
    ensemble_weights: tuple = (0., .5, 1.)

    def __post_init__(self):
        if type(self.max_native_models) is not int or not 232 <= self.max_native_models <= 256:
            raise ValueError("Complete experiment requires 232 native models; cap must be 232..256")
        if tuple(self.ensemble_weights) != (0., .5, 1.):
            raise ValueError("Preregistered pooled/seasonal-naive weight grid is 0, 0.5, 1")
        if Path(self.boosting_config).is_absolute() or ".." in Path(self.boosting_config).parts:
            raise ValueError("Boosting config must be relative to project root")


def load_config(path):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"strategies"}:
        raise ValueError("Expected [strategies] configuration")
    return StrategyConfig(**raw["strategies"])
