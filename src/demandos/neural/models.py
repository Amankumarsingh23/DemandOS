"""Controlled encoder comparison, finite BPTT, and portable state-dict checkpoints."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import NeuralConfig
from .sequences import STATIC, PAST, FUTURE, Preprocessor


def seed_run(config):
    if config.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA explicitly requested but unavailable")
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class RecurrentForecaster(nn.Module):
    def __init__(self, architecture, processor, config):
        super().__init__()
        if architecture not in ("lstm", "gru"):
            raise ValueError("Only LSTM and GRU encoders supported")
        self.architecture, self.config = architecture, config
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(processor.vocabulary[name])+1, config.embedding_dim, padding_idx=0)
            for name in STATIC])
        recurrent = nn.LSTM if architecture == "lstm" else nn.GRU
        self.encoder = recurrent(len(PAST), config.hidden_size, num_layers=1, batch_first=True, dropout=0.)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size+len(STATIC)*config.embedding_dim+len(FUTURE), config.head_size),
            nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(config.head_size, 1), nn.Softplus())

    def forward(self, past, future, categories):
        if past.ndim != 3 or past.shape[1:] != (self.config.lookback, len(PAST)):
            raise ValueError("Invalid batch-first historical tensor")
        if future.shape != (past.shape[0], self.config.horizon, len(FUTURE)) or categories.shape != (past.shape[0], len(STATIC)):
            raise ValueError("Invalid future/static tensor shapes")
        encoded, _ = self.encoder(past)
        static = torch.cat([layer(categories[:, i]) for i, layer in enumerate(self.embeddings)], dim=1)
        context = torch.cat([encoded[:, -1], static], dim=1).unsqueeze(1).expand(-1, self.config.horizon, -1)
        return self.head(torch.cat([context, future], dim=2)).squeeze(-1)


def clip_gradients(model, clip_norm):
    norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite gradient norm")
    return float(norm)


def loader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(config.seed), pin_memory=False)


def loss_epoch(model, dataset, config, optimizer=None):
    model.train(optimizer is not None)
    total, count, max_norm = 0., 0, 0.
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader(dataset, config, shuffle=optimizer is not None):
            past, future, ids, _, y = [t.to(config.device) for t in batch]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(past, future, ids)
            loss = nn.functional.mse_loss(output, y)
            if not torch.isfinite(loss) or not torch.isfinite(output).all():
                raise ValueError("Nonfinite neural loss/output; no silent fallback")
            if optimizer is not None:
                loss.backward()
                max_norm = max(max_norm, clip_gradients(model, config.clip_norm))
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError("Nonfinite model parameters after optimizer step")
            total += float(loss.detach())*len(y); count += len(y)
    return total/count, max_norm


def train(architecture, processor, training, config, *, cutoff, validation=None, epochs=None):
    """December validation selects epochs; later refits use the frozen count only."""
    cutoff = pd.Timestamp(cutoff)
    if len(training.tensors) != 5 or pd.Timestamp(processor.fit_cutoff) != cutoff:
        raise ValueError("Labelled training and cutoff-matched preprocessing required")
    if (pd.to_datetime(training.rows.origin_date)+pd.Timedelta(days=config.horizon) > cutoff).any():
        raise ValueError("Training labels exceed cutoff")
    if validation is not None:
        if len(validation.tensors) != 5 or not pd.to_datetime(validation.rows.origin_date).ge(cutoff).all():
            raise ValueError("Chronological labelled validation must follow fitting targets")
    budget = config.epochs if epochs is None else epochs
    if type(budget) is not int or not 1 <= budget <= config.epochs:
        raise ValueError("Refit epochs must remain within the predeclared budget")
    seed_run(config)
    model = RecurrentForecaster(architecture, processor, config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, best_epoch, best_weights, stale = float("inf"), 0, None, 0
    curves, started = [], perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(1, budget+1):
            start = perf_counter()
            fitting_loss, gradient = loss_epoch(model, training, config, optimizer)
            validation_loss = loss_epoch(model, validation, config)[0] if validation is not None else None
            curves.append({"epoch": epoch, "training_loss": fitting_loss, "validation_loss": validation_loss,
                           "max_gradient_norm_before_clipping": gradient, "seconds": perf_counter()-start})
            if validation is not None:
                if validation_loss < best_loss:
                    best_loss, best_epoch, stale = validation_loss, epoch, 0
                    best_weights = deepcopy(model.state_dict())
                else:
                    stale += 1
            print(f"  {architecture} epoch {epoch}/{budget}: train={fitting_loss:.5f}" +
                  (f" val={validation_loss:.5f}" if validation is not None else ""), flush=True)
            if validation is not None and stale >= config.patience:
                break
    if validation is not None:
        model.load_state_dict(best_weights)
    else:
        best_epoch = budget
    model.eval()
    metadata = {"architecture": architecture, "config": asdict(config), "fit_cutoff": str(cutoff.date()),
                "fit_windows": len(training), "fit_series": len(training.rows[["store_id", "item_id"]].drop_duplicates()),
                "fit_origin_min": str(training.rows.origin_date.min().date()), "fit_origin_max": str(training.rows.origin_date.max().date()),
                "fit_target_max": str((training.rows.origin_date.max()+pd.Timedelta(days=config.horizon)).date()),
                "validation_windows": len(validation) if validation is not None else 0,
                "validation_target_max": str((validation.rows.origin_date.max()+pd.Timedelta(days=config.horizon)).date()) if validation is not None else None,
                "epochs_run": len(curves), "selected_epochs": best_epoch,
                "best_validation_loss": best_loss if validation is not None else None,
                "parameter_count": sum(p.numel() for p in model.parameters()), "fit_seconds": perf_counter()-started,
                "warnings": [str(w.message) for w in caught], "mixed_precision": False,
                "loss": "mean squared error in per-origin history-mean-scaled units",
                "nonnegative_policy": "softplus on normalized outputs, then multiply history-only scale",
                "curves": curves, "past_features": list(PAST), "future_features": list(FUTURE)}
    return model, metadata


def predict(model, dataset, config):
    if len(dataset.tensors) != 4:
        raise ValueError("Prediction requires tensors without future labels")
    model.eval()
    outputs = []
    with torch.inference_mode():
        for past, future, ids, scale in loader(dataset, config):
            output = model(past.to(config.device), future.to(config.device), ids.to(config.device))
            outputs.append((output*scale.to(config.device)[:, None]).cpu().numpy())
    result = np.concatenate(outputs)
    if result.shape != (len(dataset), config.horizon) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Nonfinite, negative or misaligned neural predictions")
    return result


def save(model, processor, metadata, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, directory/"weights.pt")
    (directory/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (directory/"preprocessing.json").write_text(json.dumps(processor.state(), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load(directory, *, device="cpu"):
    directory = Path(directory)
    meta = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    processor = Preprocessor(**json.loads((directory/"preprocessing.json").read_text(encoding="utf-8")))
    config = NeuralConfig(**{**meta["config"], "device": device})
    seed_run(config)
    model = RecurrentForecaster(meta["architecture"], processor, config).to(device)
    model.load_state_dict(torch.load(directory/"weights.pt", map_location=device, weights_only=True))
    model.eval()
    return model, processor, meta
