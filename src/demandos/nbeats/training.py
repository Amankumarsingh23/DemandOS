"""Bounded forecast-only optimization and native PyTorch checkpoint round trips."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import torch

from demandos.neural.models import seed_run, loader, clip_gradients
from .config import NBeatsConfig
from .dataset import SCALING
from .models import NBeats


def epoch_loss(model, dataset, config, optimizer=None):
    model.train(optimizer is not None)
    total, count, maximum = 0., 0, 0.
    with torch.set_grad_enabled(optimizer is not None):
        for x, mask, _, y in loader(dataset, config, shuffle=optimizer is not None):
            x, mask, y = x.to(config.device), mask.to(config.device), y.to(config.device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(x, mask)
            loss = torch.nn.functional.mse_loss(output, y)
            if not torch.isfinite(output).all() or not torch.isfinite(loss):
                raise ValueError("Nonfinite N-BEATS output/loss; no fallback")
            if optimizer is not None:
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None and name not in model.expected_unused_parameters():
                        raise ValueError(f"Unexpected missing gradient: {name}")
                maximum = max(maximum, clip_gradients(model, config.clip_norm))
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError("Nonfinite N-BEATS parameters")
            total += float(loss.detach())*len(y); count += len(y)
    return total/count, maximum


def train(variant, training, config, *, cutoff, validation=None, validation_cutoff=None, epochs=None):
    cutoff = pd.Timestamp(cutoff)
    if len(training.tensors) != 4 or training.label_cutoff != cutoff:
        raise ValueError("Explicit fitting label cutoff must match the labelled dataset")
    if (pd.to_datetime(training.rows.origin_date)+pd.Timedelta(days=config.horizon) > cutoff).any():
        raise ValueError("Incomplete training target vectors")
    if validation is not None:
        if (validation.label_cutoff is None or validation.label_cutoff != pd.Timestamp(validation_cutoff)
                or not pd.to_datetime(validation.rows.origin_date).ge(cutoff).all()
                or (pd.to_datetime(validation.rows.origin_date)+pd.Timedelta(days=config.horizon) > pd.Timestamp(validation_cutoff)).any()):
            raise ValueError("Chronological validation and explicit target cutoff required")
    budget = config.epochs if epochs is None else epochs
    if type(budget) is not int or not 1 <= budget <= config.epochs:
        raise ValueError("Frozen refit epochs must be within the original budget")
    seed_run(config)
    model = NBeats(variant, config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    curves, best, selected, weights, stale = [], float("inf"), budget, None, 0
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(1, budget+1):
            tick = perf_counter()
            loss, gradient = epoch_loss(model, training, config, optimizer)
            valid = epoch_loss(model, validation, config)[0] if validation is not None else None
            curves.append({"epoch": epoch, "training_loss": loss, "validation_loss": valid,
                           "max_gradient_norm_before_clipping": gradient, "seconds": perf_counter()-tick})
            if validation is not None:
                if valid < best:
                    best, selected, weights, stale = valid, epoch, deepcopy(model.state_dict()), 0
                else:
                    stale += 1
            print(f"  {variant} {epoch}/{budget}: train={loss:.6f}"+(f" validation={valid:.6f}" if valid is not None else ""), flush=True)
            if validation is not None and stale >= config.patience:
                break
    if weights is not None:
        model.load_state_dict(weights)
    model.eval()
    model.fit_cutoff = cutoff
    metadata = {"variant": variant, "config": asdict(config), "scaling": SCALING, "fit_cutoff": str(cutoff.date()),
                "fit_windows": len(training), "fit_series": len(training.rows[["store_id", "item_id"]].drop_duplicates()),
                "fit_origin_min": str(training.rows.origin_date.min().date()), "fit_origin_max": str(training.rows.origin_date.max().date()),
                "fit_target_max": str((training.rows.origin_date.max()+pd.Timedelta(days=config.horizon)).date()),
                "validation_windows": len(validation) if validation is not None else 0,
                "validation_cutoff": str(pd.Timestamp(validation_cutoff).date()) if validation is not None else None,
                "epochs_run": len(curves), "selected_epochs": selected, "best_validation_loss": best if validation is not None else None,
                "parameter_count": sum(p.numel() for p in model.parameters()), "fit_seconds": perf_counter()-started,
                "block_kinds": [b.kind for b in model.blocks], "unique_blocks": len({id(b) for b in model.blocks}),
                "expected_unsupervised_final_backcast_parameters": sorted(model.expected_unused_parameters()),
                "loss": "unweighted mean squared error of raw normalized future forecasts only; no backcast loss",
                "nonnegative_policy": "train raw additive forecast; inverse-scale then clip at zero for scoring; retain raw/components",
                "warnings": [str(w.message) for w in caught], "torch_version": torch.__version__, "curves": curves}
    return model, metadata


def predict(model, dataset, config):
    if len(dataset.tensors) != 3 or dataset.label_cutoff is not None:
        raise ValueError("Inference accepts history/mask/scale without future targets")
    if not hasattr(model, "fit_cutoff") or pd.to_datetime(dataset.rows.origin_date).min() < model.fit_cutoff:
        raise ValueError("A fitted model cannot predict before its fitting cutoff")
    model.eval()
    total, components = [], []
    with torch.inference_mode():
        for x, mask, scale in loader(dataset, config):
            result = model(x.to(config.device), mask.to(config.device), diagnostics=True)
            raw = result["total"].cpu().numpy().astype(float)*scale.numpy()[:, None]
            parts = result["contributions"].cpu().numpy().astype(float)*scale.numpy()[:, None, None]
            np.testing.assert_allclose(raw, parts.sum(axis=1), rtol=1e-6, atol=1e-4)
            total.append(raw); components.append(parts)
    raw, parts = np.concatenate(total), np.concatenate(components)
    if raw.shape != (len(dataset), config.horizon) or not np.isfinite(raw).all() or not np.isfinite(parts).all():
        raise ValueError("Nonfinite/misaligned N-BEATS forecast")
    return raw, np.maximum(raw, 0), parts


def save(model, metadata, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory/"weights.pt")
    (directory/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load(directory, *, device="cpu"):
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    config = replace(NBeatsConfig(**metadata["config"]), device=device)
    if metadata["scaling"] != SCALING:
        raise ValueError("Checkpoint normalization contract changed")
    cutoff = pd.Timestamp(metadata.get("fit_cutoff"))
    target_max = pd.Timestamp(metadata.get("fit_target_max"))
    if pd.isna(cutoff) or pd.isna(target_max) or cutoff != cutoff.normalize() or target_max > cutoff:
        raise ValueError("Checkpoint fitting/target cutoffs are missing or invalid")
    seed_run(config)
    model = NBeats(metadata["variant"], config).to(device)
    model.load_state_dict(torch.load(directory/"weights.pt", weights_only=True, map_location=device))
    model.fit_cutoff = cutoff
    return model.eval(), metadata
