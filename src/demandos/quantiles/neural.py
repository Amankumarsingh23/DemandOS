"""Generic N-BEATS-based residual network with a monotone joint quantile head."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

from demandos.nbeats.models import NBeats
from demandos.nbeats.config import NBeatsConfig
from demandos.nbeats.dataset import SCALING
from demandos.neural.models import seed_run, loader, clip_gradients
from .contracts import LEVELS


def joint_loss(actual, quantiles):
    if (quantiles.shape != (*actual.shape, 3) or not torch.isfinite(actual).all()
            or not torch.isfinite(quantiles).all() or (actual < 0).any()):
        raise ValueError("Finite aligned observed targets and [B,28,3] quantiles required")
    tau = quantiles.new_tensor(LEVELS)
    error = actual.unsqueeze(-1)-quantiles
    return torch.maximum(tau*error, (tau-1)*error).mean()


class QuantileNBeats(NBeats):
    def __init__(self, config):
        super().__init__("generic", config)
        for block in set(self.blocks):
            block.forecast_basis = nn.Linear(config.generic_theta, config.horizon*3, bias=False)

    def forward(self, x, mask, *, diagnostics=False):
        if (x.ndim != 2 or x.shape[1] != self.config.lookback or mask.shape != x.shape or len(x) == 0
                or not torch.isfinite(x).all() or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all()):
            raise ValueError("Finite history/binary mask [B,56] required")
        residual, latent = x*mask, x.new_zeros((len(x), self.config.horizon, 3))
        residuals, backcasts, parts = [residual], [], []
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = (residual-backcast)*mask
            contribution = forecast.reshape(len(x), self.config.horizon, 3)
            latent = latent+contribution
            if diagnostics:
                residuals.append(residual); backcasts.append(backcast); parts.append(contribution)
        # Positive first quantile and positive increments: exact zero ties are not imposed.
        quantiles = torch.cumsum(torch.nn.functional.softplus(latent), dim=-1)
        if diagnostics:
            return {"quantiles": quantiles, "latent": latent, "contributions": torch.stack(parts, 1),
                    "residuals": torch.stack(residuals, 1), "backcasts": torch.stack(backcasts, 1)}
        return quantiles


def epoch(model, dataset, config, optimizer=None):
    model.train(optimizer is not None)
    total, count, maximum = 0., 0, 0.
    with torch.set_grad_enabled(optimizer is not None):
        for x, mask, _, y in loader(dataset, config, shuffle=optimizer is not None):
            x, mask, y = x.to(config.device), mask.to(config.device), y.to(config.device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = joint_loss(y, model(x, mask))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite joint pinball loss")
            if optimizer is not None:
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None and name not in model.expected_unused_parameters():
                        raise ValueError(f"Missing quantile gradient: {name}")
                maximum = max(maximum, clip_gradients(model, config.clip_norm))
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError("Nonfinite quantile parameter")
            total += float(loss.detach())*len(y); count += len(y)
    return total/count, maximum


def train(training, config, *, cutoff, validation=None, validation_cutoff=None, epochs=None):
    cutoff = pd.Timestamp(cutoff)
    if len(training.tensors) != 4 or training.label_cutoff != cutoff:
        raise ValueError("Explicit matching complete-vector training cutoff required")
    if (pd.to_datetime(training.rows.origin_date)+pd.Timedelta(days=28) > cutoff).any():
        raise ValueError("Future training target vector")
    if validation is not None:
        if (validation.label_cutoff != pd.Timestamp(validation_cutoff)
                or not pd.to_datetime(validation.rows.origin_date).ge(cutoff).all()
                or (pd.to_datetime(validation.rows.origin_date)+pd.Timedelta(days=28) > pd.Timestamp(validation_cutoff)).any()):
            raise ValueError("Chronological validation with complete labels required")
    budget = config.epochs if epochs is None else epochs
    if type(budget) is not int or not 1 <= budget <= config.epochs:
        raise ValueError("Invalid frozen epoch budget")
    seed_run(config)
    model = QuantileNBeats(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    curves, best, weights, selected, stale = [], float("inf"), None, budget, 0
    began = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for step in range(1, budget+1):
            tick = perf_counter()
            loss, gradient = epoch(model, training, config, optimizer)
            value = epoch(model, validation, config)[0] if validation is not None else None
            curves.append({"epoch": step, "training_pinball_normalized": loss, "validation_pinball_normalized": value,
                           "max_gradient_before_clip": gradient, "seconds": perf_counter()-tick})
            if validation is not None:
                if value < best:
                    best, selected, weights, stale = value, step, deepcopy(model.state_dict()), 0
                else:
                    stale += 1
            print(f"  joint quantiles {step}/{budget}: train={loss:.6f}"+(f" validation={value:.6f}" if value is not None else ""), flush=True)
            if validation is not None and stale >= config.patience:
                break
    if weights is not None:
        model.load_state_dict(weights)
    model.fit_cutoff = cutoff
    model.fit_target_max = training.rows.origin_date.max()+pd.Timedelta(days=28)
    meta = {"architecture": "generic_nbeats_based_monotone_joint_quantiles", "config": asdict(config),
            "levels": list(LEVELS), "scaling": SCALING, "fit_cutoff": str(cutoff.date()),
            "fit_target_max": str((training.rows.origin_date.max()+pd.Timedelta(days=28)).date()),
            "fit_windows": len(training), "training_series": len(training.rows[["store_id", "item_id"]].drop_duplicates()),
            "fit_origin_min": str(training.rows.origin_date.min().date()), "fit_origin_max": str(training.rows.origin_date.max().date()),
            "validation_windows": len(validation) if validation is not None else 0,
            "validation_cutoff": str(pd.Timestamp(validation_cutoff).date()) if validation is not None else None,
            "epochs_run": len(curves), "selected_epochs": selected,
            "best_validation_pinball_normalized": best if validation is not None else None,
            "parameters": sum(p.numel() for p in model.parameters()), "fit_seconds": perf_counter()-began,
            "warnings": [str(w.message) for w in caught], "curves": curves,
            "zero_policy": "softplus first value/increments approach zero but exact zero ties are not parameterized",
            "loss": "mean pinball across normalized targets, horizons and three quantiles"}
    return model.eval(), meta


def predict(model, dataset, config):
    if len(dataset.tensors) != 3 or dataset.label_cutoff is not None:
        raise ValueError("Inference must contain only history, mask and own-origin scale")
    if not hasattr(model, "fit_cutoff") or pd.to_datetime(dataset.rows.origin_date).min() < model.fit_cutoff:
        raise ValueError("Future-fitted quantile network")
    model.eval()
    results = []
    with torch.inference_mode():
        for x, mask, scale in loader(dataset, config):
            values = model(x.to(config.device), mask.to(config.device)).cpu().numpy().astype(float)
            results.append(values*scale.numpy()[:, None, None])
    result = np.concatenate(results)
    if result.shape != (len(dataset), 28, 3) or not np.isfinite(result).all() or (result < 0).any() or (np.diff(result, axis=-1) < 0).any():
        raise ValueError("Invalid joint quantile output")
    return result


def save(model, metadata, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory/"weights.pt")
    (directory/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text())
    if metadata["levels"] != list(LEVELS) or metadata["scaling"] != SCALING or metadata["architecture"] != "generic_nbeats_based_monotone_joint_quantiles":
        raise ValueError("Quantile checkpoint architecture/scaling contract mismatch")
    config = NBeatsConfig(**metadata["config"])
    cutoff, target = pd.Timestamp(metadata.get("fit_cutoff")), pd.Timestamp(metadata.get("fit_target_max"))
    if pd.isna(cutoff) or pd.isna(target) or target > cutoff:
        raise ValueError("Missing/future neural fitting labels")
    seed_run(config)
    model = QuantileNBeats(config).to(config.device)
    model.load_state_dict(torch.load(directory/"weights.pt", weights_only=True, map_location=config.device))
    model.fit_cutoff = cutoff
    model.fit_target_max = target
    return model.eval(), metadata
