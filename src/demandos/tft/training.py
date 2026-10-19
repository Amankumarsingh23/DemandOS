"""Finite, chronological point-MSE training and versioned preprocessing checkpoints."""

from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import torch

from demandos.neural.models import seed_run,loader,clip_gradients
from demandos.neural.sequences import Preprocessor,validate_examples,PAST,FUTURE,STATIC
from .config import TFTConfig
from .model import TemporalFusionTransformer


SCHEMA_VERSION = 1


def epoch_loss(model,dataset,config,optimizer=None):
    model.train(optimizer is not None)
    total,count,maximum=0.,0,0.
    with torch.set_grad_enabled(optimizer is not None):
        for past,future,ids,_,labels in loader(dataset,config,shuffle=optimizer is not None):
            past,future,ids,labels=[value.to(config.device) for value in (past,future,ids,labels)]
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            prediction=model(past,future,ids)
            loss=torch.nn.functional.mse_loss(prediction,labels)
            if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite TFT output/loss; no fallback")
            if optimizer is not None:
                loss.backward()
                for name,parameter in model.named_parameters():
                    if parameter.grad is None: raise ValueError(f"Missing TFT gradient: {name}")
                maximum=max(maximum,clip_gradients(model,config.clip_norm))
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()): raise ValueError("Nonfinite TFT parameters")
            total+=float(loss.detach())*len(labels); count+=len(labels)
    return total/count,maximum


def train(training,processor,config,*,cutoff,validation=None,validation_cutoff=None,epochs=None):
    validate_examples(training.rows,config,cutoff)
    if len(training.tensors)!=5 or pd.Timestamp(processor.fit_cutoff)!=pd.Timestamp(cutoff):
        raise ValueError("Labelled training with cutoff-matched preprocessing required")
    if validation is not None:
        validate_examples(validation.rows,config,validation_cutoff)
        if len(validation.tensors)!=5 or validation_cutoff is None or not pd.to_datetime(validation.rows.origin_date).ge(pd.Timestamp(cutoff)).all():
            raise ValueError("Explicit chronological validation labels required")
    budget=config.epochs if epochs is None else epochs
    if type(budget) is not int or not 1<=budget<=config.epochs: raise ValueError("Epoch cap exceeded")
    seed_run(config)
    model=TemporalFusionTransformer(processor,config).to(config.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    curves,best,selected,weights,stale=[],float("inf"),budget,None,0
    started=perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(1,budget+1):
            tick=perf_counter()
            loss,norm=epoch_loss(model,training,config,optimizer)
            valid=epoch_loss(model,validation,config)[0] if validation is not None else None
            curves.append({"epoch":epoch,"training_loss":loss,"validation_loss":valid,"max_gradient_norm_before_clipping":norm,"seconds":perf_counter()-tick})
            if validation is not None:
                if valid<best: best,selected,weights,stale=valid,epoch,deepcopy(model.state_dict()),0
                else: stale+=1
            print(f"  TFT {epoch}/{budget}: train={loss:.6f}"+(f" validation={valid:.6f}" if valid is not None else ""),flush=True)
            if validation is not None and stale>=config.patience: break
    if weights is not None: model.load_state_dict(weights)
    model.fit_cutoff=pd.Timestamp(cutoff)
    model.eval()
    meta={"schema_version":SCHEMA_VERSION,"config":asdict(config),"torch_version":torch.__version__,
          "fit_cutoff":str(pd.Timestamp(cutoff).date()),"fit_windows":len(training),
          "fit_series":len(training.rows[["store_id","item_id"]].drop_duplicates()),
          "fit_origin_min":str(training.rows.origin_date.min().date()),"fit_origin_max":str(training.rows.origin_date.max().date()),
          "fit_target_max":str((training.rows.origin_date.max()+pd.Timedelta(days=config.horizon)).date()),
          "validation_windows":len(validation) if validation is not None else 0,
          "validation_cutoff":str(pd.Timestamp(validation_cutoff).date()) if validation is not None else None,
          "epochs_run":len(curves),"selected_epochs":selected,"best_validation_loss":best if validation is not None else None,
          "parameter_count":sum(p.numel() for p in model.parameters()),"fit_seconds":perf_counter()-started,
          "past_features":list(PAST),"future_features":list(FUTURE),"static_features":list(STATIC),
          "objective":"unweighted normalized raw point-forecast MSE; no teacher forcing or quantile loss",
          "nonnegative_policy":"inverse-scale raw linear outputs; score max(0,raw); retain raw outputs",
          "warnings":[str(w.message) for w in caught],"curves":curves}
    return model,meta


def validate_diagnostics(result):
    for name in ("static_weights","past_weights","future_weights","attention"):
        values=result[name]
        if not torch.isfinite(values).all() or (values<0).any(): raise ValueError(f"Invalid {name}")
        torch.testing.assert_close(values.sum(-1),torch.ones_like(values.sum(-1)),atol=1e-6,rtol=1e-6)
    if (result["attention"].masked_select(~result["allowed"][:,None].expand_as(result["attention"]))!=0).any():
        raise ValueError("Forbidden attention key has nonzero probability")


def predict(model,dataset,config,*,diagnostics=False):
    if len(dataset.tensors)!=4: raise ValueError("Inference must not contain labels")
    if not hasattr(model,"fit_cutoff") or pd.to_datetime(dataset.rows.origin_date).min()<model.fit_cutoff:
        raise ValueError("A future-fitted TFT cannot predict an earlier origin")
    model.eval()
    predictions,diagnostic_batches=[],[]
    with torch.inference_mode():
        for past,future,ids,scale in loader(dataset,config):
            result=model(past.to(config.device),future.to(config.device),ids.to(config.device),diagnostics=diagnostics)
            if diagnostics:
                validate_diagnostics(result)
                raw=result["prediction"]
                diagnostic_batches.append({name:result[name].cpu().numpy() for name in ("static_weights","past_weights","future_weights","attention","allowed")})
            else: raw=result
            predictions.append(raw.cpu().numpy().astype(float)*scale.numpy()[:,None])
    raw=np.concatenate(predictions)
    if raw.shape!=(len(dataset),config.horizon) or not np.isfinite(raw).all(): raise ValueError("Nonfinite/misaligned TFT predictions")
    extra={name:np.concatenate([batch[name] for batch in diagnostic_batches]) for name in diagnostic_batches[0]} if diagnostics else {}
    return raw,np.maximum(raw,0),extra


def save(model,processor,metadata,directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    torch.save({name:value.detach().cpu() for name,value in model.state_dict().items()},directory/"weights.pt")
    (directory/"metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (directory/"preprocessing.json").write_text(json.dumps(processor.state(),indent=2,allow_nan=False)+"\n",encoding="utf-8")


def load(directory,*,device="cpu"):
    directory=Path(directory)
    meta=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    processor=Preprocessor(**json.loads((directory/"preprocessing.json").read_text(encoding="utf-8")))
    cutoff,target=pd.Timestamp(meta.get("fit_cutoff")),pd.Timestamp(meta.get("fit_target_max"))
    if meta.get("schema_version")!=SCHEMA_VERSION or pd.isna(cutoff) or pd.isna(target) or target>cutoff or pd.Timestamp(processor.fit_cutoff)!=cutoff:
        raise ValueError("Invalid TFT checkpoint/preprocessing cutoffs or schema version")
    if meta["past_features"]!=list(PAST) or meta["future_features"]!=list(FUTURE) or meta["static_features"]!=list(STATIC):
        raise ValueError("Checkpoint feature schema differs")
    config=replace(TFTConfig(**meta["config"]),device=device); seed_run(config)
    model=TemporalFusionTransformer(processor,config).to(device)
    model.load_state_dict(torch.load(directory/"weights.pt",map_location=device,weights_only=True))
    model.fit_cutoff=cutoff
    return model.eval(),processor,meta
