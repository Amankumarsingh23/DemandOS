"""Independent N-BEATS implementation: backward residuals and forward contributions."""

import math
import torch
from torch import nn


def trend_basis(lookback, horizon, degree):
    # Continuous signed day grid, with the observed origin at t=0.
    past = torch.arange(1-lookback, 1, dtype=torch.float32)/lookback
    future = torch.arange(1, horizon+1, dtype=torch.float32)/lookback
    return torch.stack([past**i for i in range(degree+1)]), torch.stack([future**i for i in range(degree+1)])


def fourier_basis(lookback, horizon, harmonics, period):
    def matrix(days):
        values = [torch.ones_like(days)]
        for harmonic in range(1, harmonics+1):
            phase = 2*math.pi*harmonic*days/period
            values.extend([torch.cos(phase), torch.sin(phase)])
        return torch.stack(values)
    return matrix(torch.arange(1-lookback, 1, dtype=torch.float32)), matrix(torch.arange(1, horizon+1, dtype=torch.float32))


class Block(nn.Module):
    def __init__(self, kind, config):
        super().__init__()
        self.kind = kind
        layers, width = [], config.lookback
        for _ in range(config.layers):
            layers.extend([nn.Linear(width, config.hidden_size), nn.ReLU(), nn.Dropout(config.dropout)])
            width = config.hidden_size
        self.hidden = nn.Sequential(*layers)
        if kind == "generic":
            theta = config.generic_theta
            self.backcast_basis = nn.Linear(theta, config.lookback, bias=False)
            self.forecast_basis = nn.Linear(theta, config.horizon, bias=False)
        else:
            if kind == "trend":
                backcast, forecast = trend_basis(config.lookback, config.horizon, config.trend_degree)
            elif kind == "seasonality":
                backcast, forecast = fourier_basis(config.lookback, config.horizon, config.harmonics, config.period)
            else:
                raise ValueError("Unsupported N-BEATS block kind")
            theta = len(backcast)
            self.register_buffer("backcast_basis", backcast)
            self.register_buffer("forecast_basis", forecast)
        self.theta_backcast = nn.Linear(width, theta)
        self.theta_forecast = nn.Linear(width, theta)

    def forward(self, residual):
        hidden = self.hidden(residual)
        back, future = self.theta_backcast(hidden), self.theta_forecast(hidden)
        if self.kind == "generic":
            return self.backcast_basis(back), self.forecast_basis(future)
        return back @ self.backcast_basis, future @ self.forecast_basis


class NBeats(nn.Module):
    def __init__(self, variant, config):
        super().__init__()
        self.variant, self.config = variant, config
        if variant == "generic":
            stacks = [("generic", config.generic_blocks, config.generic_share)]
        elif variant == "interpretable":
            stacks = [("trend", config.trend_blocks, config.interpretable_share),
                      ("seasonality", config.seasonality_blocks, config.interpretable_share)]
        else:
            raise ValueError("Only generic/interpretable N-BEATS supported")
        blocks = []
        for kind, count, sharing in stacks:
            shared = Block(kind, config) if sharing else None
            blocks.extend([shared if sharing else Block(kind, config) for _ in range(count)])
        self.blocks = nn.ModuleList(blocks)

    def expected_unused_parameters(self):
        # Forecast-only supervision never uses an unshared final backcast branch.
        last = self.blocks[-1]
        if sum(block is last for block in self.blocks) > 1:
            return set()
        ids = {id(p) for name, p in last.named_parameters() if name.startswith(("theta_backcast.", "backcast_basis."))}
        return {name for name, p in self.named_parameters() if id(p) in ids}

    def forward(self, x, mask, *, diagnostics=False):
        if x.ndim != 2 or x.shape[1] != self.config.lookback or mask.shape != x.shape or len(x) == 0:
            raise ValueError("Expected nonempty X/mask [B,56]")
        if not torch.isfinite(x).all() or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Finite target values and binary valid-history masks required")
        residual = x*mask
        total = x.new_zeros((len(x), self.config.horizon))
        contributions, residuals, backcasts = [], [residual], []
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = (residual-backcast)*mask
            total = total+forecast
            if diagnostics:
                contributions.append(forecast); residuals.append(residual); backcasts.append(backcast)
        if diagnostics:
            return {"total": total, "contributions": torch.stack(contributions, dim=1),
                    "residuals": torch.stack(residuals, dim=1), "backcasts": torch.stack(backcasts, dim=1)}
        return total
