"""Predeclared, finite hypothetical shifts; never actual supplier information."""
from dataclasses import dataclass
import numpy as np
from .model import Scenarios

FAMILIES = ("baseline","surge","burst","long_lead","supplier_closure","joint","shifted_probability")


@dataclass(frozen=True)
class Closure:
    start: int = 8
    end: int = 14

    def __post_init__(self):
        if type(self.start) is not int or type(self.end) is not int or not 1 <= self.start <= self.end <= 28:
            raise ValueError("Bounded integer receiving closure required")

    def __call__(self, day, date):
        # At this opening, all due orders are postponed to the first open day.
        return self.end+1-day if self.start <= day <= self.end else 0


def shifted(paths, family):
    x = np.asarray(paths)
    if family not in FAMILIES or x.shape[-1] != 28 or not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any():
        raise ValueError("Complete integer 28-day paths and known stress family required")
    x = x.copy()
    if family in {"surge","joint"}: x = np.floor(x*1.5+.5).astype(int)
    if family == "burst":
        # Shared seven-day amplification changes cross-series dependence.
        x[...,7:14] *= 3
    return x


def planning(paths, family="baseline", shock_probability=.1):
    """Four normal and four adverse joint paths; shared shock-family indicator.

    Prefix-sampled normal 14-day demand vs assumed 1.5x, 21-day shock demand.
    Half-up integer rounding. This finite support is not a full box counterpart.
    """
    if family not in FAMILIES or not np.isfinite(shock_probability) or not 0 <= shock_probability <= 1:
        raise ValueError("Invalid planning family/probability")
    paths = shifted(paths, "baseline")
    if paths.ndim != 3 or paths.shape[1] < 8: raise ValueError("Eight joint historical planning paths required")
    normal = paths[:, :4, :14].sum(axis=2)
    shock = np.floor(paths[:,4:8,:21]*1.5+.5).astype(int).sum(axis=2)
    demand = np.concatenate([normal,shock],axis=1)
    p = .2 if family == "shifted_probability" else shock_probability
    return Scenarios(demand,np.r_[np.full(4,(1-p)/4),np.full(4,p/4)],
                     tuple([f"normal_{i}" for i in range(4)]+[f"shock_{i}" for i in range(4)]))


def uncertainty_box(scenarios):
    """Diagnostic bounds containing the support, not optimized box vertices."""
    return dict(lower=scenarios.demand.min(axis=1).tolist(),upper=scenarios.demand.max(axis=1).tolist(),
                interpretation="coordinate bounds only; optimizer uses declared finite joint support; no Gamma probability")
