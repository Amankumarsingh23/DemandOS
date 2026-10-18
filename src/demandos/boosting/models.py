"""Consistent native CPU fit/predict/save/load APIs with strict temporal contracts."""

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from importlib import import_module
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import re
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from demandos.features.config import FeatureConfig
from demandos.features.contracts import KEYS, predictive_columns
from demandos.features.encoding import CATEGORICAL
from .config import BoostConfig, LIBRARIES


TARGET = "target_units_sold"
UNKNOWN = "__DEMANDOS_UNKNOWN__"


def validate_rows(frame, *, labelled=False, cutoff=None):
    required = KEYS + ["target_date", "max_sales_source_date", "mapping_fit_end"] + ([TARGET] if labelled else [])
    if frame.empty or not set(required).issubset(frame) or frame[required].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Nonempty unique feature/label keys and provenance required")
    origins, targets = pd.to_datetime(frame.origin_date), pd.to_datetime(frame.target_date)
    if not origins.eq(origins.dt.normalize()).all() or not targets.eq(targets.dt.normalize()).all():
        raise ValueError("Daily midnight origin and target dates required")
    if not frame.horizon.isin(range(1, 29)).all() or not targets.eq(origins + pd.to_timedelta(frame.horizon.astype(int), unit="D")).all():
        raise ValueError("Invalid direct forecast horizon/date alignment")
    if not pd.to_datetime(frame.max_sales_source_date).eq(origins).all() or not pd.to_datetime(frame.mapping_fit_end).le(origins).all():
        raise ValueError("Future source observations or future-fitted features")
    for name in frame:
        match = re.fullmatch(r"origin_lag_(\d+)", name)
        if match:
            source = f"{name}_source_date"
            if source not in frame or not frame[name].isna().eq(frame[source].isna()).all():
                raise ValueError("Missing lag timestamp provenance")
            known = frame[name].notna()
            if not pd.to_datetime(frame.loc[known, source]).eq(origins[known] - pd.Timedelta(days=int(match[1]))).all():
                raise ValueError("Target-relative or future lag source")
    for prefix in ("last", "prior"):
        price = f"{prefix}_known_price"
        if price in frame:
            known = frame[price].notna()
            for suffix in ("observed_at", "available_at"):
                stamps = pd.to_datetime(frame[f"{prefix}_price_{suffix}"])
                if stamps[known].isna().any() or not stamps[known].le(origins[known]).all():
                    raise ValueError("Historical price violates origin availability")
    if labelled:
        observed = frame[TARGET].to_numpy(float)
        if not np.isfinite(observed).all() or (observed < 0).any() or (observed != np.floor(observed)).any():
            raise ValueError("Invalid observed-sales labels")
        if cutoff is None or pd.isna(pd.Timestamp(cutoff)) or not targets.le(pd.Timestamp(cutoff)).all():
            raise ValueError("Every training/validation target must be observed by its allowed cutoff")
    elif TARGET in frame or "actual" in frame:
        raise ValueError("Prediction accepts inference-only features, without actual labels")


class GlobalModel:
    def __init__(self, library, features, config=BoostConfig(), feature_config=FeatureConfig()):
        allowed = set(predictive_columns(feature_config)) - {f"{name}_code" for name in CATEGORICAL}
        if library not in LIBRARIES or not features or len(set(features)) != len(features) or set(features) - allowed or "horizon" not in features:
            raise ValueError("Explicit unique origin-safe predictors including horizon required; labels/audits/codes forbidden")
        self.library, self.features, self.config, self.feature_config = library, list(features), config, feature_config
        self.categories = [name for name in CATEGORICAL if name in features]
        self.vocabulary, self.booster, self.metadata = {}, None, {}

    def matrix(self, frame):
        if not set(self.features).issubset(frame):
            raise ValueError("Inference predictor schema does not match fitted features")
        output = pd.DataFrame(index=frame.index)
        for name in self.features:
            if name in self.categories:
                values = frame[name].astype("string").fillna(UNKNOWN)
                values = values.where(values.isin(self.vocabulary[name]), UNKNOWN)
                output[name] = (values.astype(str) if self.library == "catboost" else
                                pd.Categorical(values, categories=[UNKNOWN, *self.vocabulary[name]], ordered=False))
            else:
                values = pd.to_numeric(frame[name], errors="raise").astype("float32")
                if np.isinf(values.to_numpy()).any():
                    raise ValueError(f"Infinite numerical feature: {name}")
                output[name] = values
        return output

    def fit(self, training, *, allowed_cutoff, validation=None, validation_cutoff=None):
        validate_rows(training, labelled=True, cutoff=allowed_cutoff)
        if len(training) > self.config.max_rows:
            raise ValueError("Training row cap exceeded")
        if training[["store_id", "item_id"]].drop_duplicates().shape[0] > self.config.training_series:
            raise ValueError("Training series cap exceeded")
        if training.origin_date.nunique() > self.config.max_origins:
            raise ValueError("Training origin cap exceeded")
        if validation is not None:
            validate_rows(validation, labelled=True, cutoff=validation_cutoff)
            if not pd.to_datetime(validation.target_date).gt(pd.Timestamp(allowed_cutoff)).all() or not pd.to_datetime(validation.origin_date).ge(pd.Timestamp(allowed_cutoff)).all():
                raise ValueError("Chronological early-stopping validation must follow the fitting label cutoff")
            if len(validation) > self.config.max_rows:
                raise ValueError("Validation row cap exceeded")
        # Native category vocabularies use FIT rows only, never validation/test categories.
        for name in self.categories:
            if name not in training:
                raise ValueError("Missing categorical predictor")
            values = training[name].dropna().astype(str)
            if UNKNOWN in set(values):
                raise ValueError("Reserved unknown-category token appears in actual identifiers")
            self.vocabulary[name] = sorted(values.unique())
        order = ["target_date", "origin_date", "store_id", "item_id", "horizon"]
        train = training.sort_values(order).reset_index(drop=True)
        valid = validation.sort_values(order).reset_index(drop=True) if validation is not None else None
        X, y = self.matrix(train), train[TARGET].to_numpy(float)
        Xv, yv = (self.matrix(valid), valid[TARGET].to_numpy(float)) if valid is not None else (None, None)
        started, output = perf_counter(), StringIO()
        caught = []
        with warnings.catch_warnings(record=True) as caught, redirect_stdout(output), redirect_stderr(output):
            warnings.simplefilter("always")
            self._fit_native(X, y, Xv, yv)
        self.metadata = {
            "library": self.library, "version": version(self.library), "features": self.features,
            "categorical_features": self.categories, "categorical_vocabulary_fit": "training-eligible fit rows only",
            "allowed_cutoff": str(pd.Timestamp(allowed_cutoff).date()),
            "training_target_min": str(pd.Timestamp(train.target_date.min()).date()),
            "training_target_max": str(pd.Timestamp(train.target_date.max()).date()),
            "training_origin_min": str(pd.Timestamp(train.origin_date.min()).date()),
            "training_origin_max": str(pd.Timestamp(train.origin_date.max()).date()),
            "training_rows": len(train), "training_series": len(train[["store_id", "item_id"]].drop_duplicates()),
            "training_origins": train.origin_date.nunique(), "validation_rows": len(valid) if valid is not None else 0,
            "validation_cutoff": str(pd.Timestamp(validation_cutoff).date()) if validation is not None else None,
            "validation_target_min": str(pd.Timestamp(valid.target_date.min()).date()) if valid is not None else None,
            "validation_target_max": str(pd.Timestamp(valid.target_date.max()).date()) if valid is not None else None,
            "rounds": self.rounds, "max_rounds": self.config.iterations, "parameters": self.native_parameters,
            "fit_seconds": perf_counter() - started,
            "warnings": [f"{type(w.message).__name__}: {w.message}" for w in caught],
            "native_warning_lines": sorted(set(line for line in output.getvalue().splitlines() if "[Warning]" in line)),
            "fit_log": output.getvalue(), "nonnegative_policy": "preserve raw; score max(0,raw); never change actuals"}
        return self

    def _fit_native(self, X, y, Xv, yv):
        c, native = self.config, import_module(self.library)
        if self.library == "lightgbm":
            params = {"objective": "regression_l2", "metric": "rmse", "learning_rate": c.learning_rate,
                      "num_leaves": c.num_leaves, "max_depth": c.max_depth, "min_child_samples": c.min_child_samples,
                      "lambda_l1": c.reg_alpha, "lambda_l2": c.reg_lambda, "feature_fraction": c.colsample,
                      "bagging_fraction": c.subsample, "bagging_freq": 1, "num_threads": c.threads,
                      "seed": c.seed, "deterministic": True, "force_col_wise": True, "verbosity": 1}
            data = native.Dataset(X, label=y, categorical_feature=self.categories)
            valid = native.Dataset(Xv, label=yv, reference=data, categorical_feature=self.categories) if Xv is not None else None
            self.booster = native.train(params, data, num_boost_round=c.iterations,
                valid_sets=[valid] if valid is not None else None,
                callbacks=[native.early_stopping(c.patience, verbose=False)] if valid is not None else [])
            self.rounds = self.booster.best_iteration or self.booster.current_iteration()
        elif self.library == "xgboost":
            params = {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist", "device": "cpu",
                      "eta": c.learning_rate, "max_depth": c.max_depth, "min_child_weight": c.min_child_weight,
                      "alpha": c.reg_alpha, "lambda": c.reg_lambda, "subsample": c.subsample,
                      "colsample_bytree": c.colsample, "nthread": c.threads, "seed": c.seed, "max_bin": 128}
            data = native.DMatrix(X, label=y, enable_categorical=True, nthread=c.threads)
            valid = native.DMatrix(Xv, label=yv, enable_categorical=True, nthread=c.threads) if Xv is not None else None
            self.booster = native.train(params, data, num_boost_round=c.iterations,
                evals=[(valid, "validation")] if valid is not None else [],
                early_stopping_rounds=c.patience if valid is not None else None, verbose_eval=False)
            self.rounds = self.booster.best_iteration + 1 if valid is not None else self.booster.num_boosted_rounds()
        else:
            params = {"iterations": c.iterations, "depth": c.max_depth, "learning_rate": c.learning_rate,
                      "l2_leaf_reg": c.cat_l2, "loss_function": "RMSE", "eval_metric": "RMSE",
                      "boosting_type": "Plain", "has_time": True, "task_type": "CPU", "thread_count": c.threads,
                      "random_seed": c.seed, "allow_writing_files": False, "verbose": False,
                      "bootstrap_type": "Bernoulli", "subsample": c.subsample}
            self.booster = native.CatBoostRegressor(**params)
            self.booster.fit(X, y, cat_features=self.categories, eval_set=(Xv, yv) if Xv is not None else None,
                             early_stopping_rounds=c.patience if Xv is not None else None, use_best_model=Xv is not None)
            self.rounds = self.booster.tree_count_
        self.native_parameters = params

    def predict(self, features):
        if self.booster is None:
            raise ValueError("Model must be fitted or loaded")
        validate_rows(features)
        if not pd.to_datetime(features.origin_date).ge(pd.Timestamp(self.metadata["allowed_cutoff"])).all():
            raise ValueError("A model trained on later labels cannot predict an earlier origin")
        X = self.matrix(features)
        if self.library == "lightgbm":
            raw = self.booster.predict(X, num_iteration=self.rounds, validate_features=True, num_threads=self.config.threads)
        elif self.library == "xgboost":
            native = import_module(self.library)
            raw = self.booster.predict(native.DMatrix(X, enable_categorical=True, nthread=self.config.threads), iteration_range=(0, self.rounds))
        else:
            raw = self.booster.predict(X, ntree_end=self.rounds, thread_count=self.config.threads)
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (len(features),) or not np.isfinite(raw).all():
            raise ValueError("Nonfinite or misaligned model predictions; no fallback")
        result = features[KEYS + ["target_date"]].reset_index(drop=True).copy()
        result["model"], result["raw_prediction"], result["prediction"] = self.library, raw, np.maximum(raw, 0)
        return result

    def importance(self):
        if self.library == "lightgbm":
            importance = self.booster.feature_importance("gain", iteration=self.rounds)
            splits, kind = self.booster.feature_importance("split", iteration=self.rounds), "gain"
        elif self.library == "xgboost":
            selected = self.booster[:self.rounds]
            gains, weights = selected.get_score(importance_type="total_gain"), selected.get_score(importance_type="weight")
            importance, splits, kind = [gains.get(f, 0) for f in self.features], [weights.get(f, 0) for f in self.features], "total_gain"
        else:
            importance, splits, kind = self.booster.get_feature_importance(), [None] * len(self.features), "PredictionValuesChange"
        return pd.DataFrame({"model": self.library, "feature": self.features, "importance": importance,
                             "splits": splits, "kind": kind}).sort_values(["importance", "feature"], ascending=[False, True])

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = {"lightgbm": "model.txt", "xgboost": "model.ubj", "catboost": "model.cbm"}[self.library]
        if self.library == "lightgbm":
            self.booster.save_model(str(directory / name), num_iteration=self.rounds)
        else:
            self.booster.save_model(str(directory / name))
        payload = {"library": self.library, "features": self.features, "config": asdict(self.config),
                   "feature_config": asdict(self.feature_config), "vocabulary": self.vocabulary,
                   "rounds": self.rounds, "native_parameters": self.native_parameters, "metadata": self.metadata, "model_file": name}
        (directory / "metadata.json").write_text(json.dumps(payload, default=str, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        raw = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        model = cls(raw["library"], raw["features"], BoostConfig(**raw["config"]), FeatureConfig(**raw["feature_config"]))
        native, path = import_module(model.library), directory / raw["model_file"]
        if path.parent.resolve() != directory.resolve():
            raise ValueError("Model file must be local to its artifact directory")
        if model.library == "lightgbm":
            model.booster = native.Booster(model_file=str(path))
        elif model.library == "xgboost":
            model.booster = native.Booster(params={"nthread": model.config.threads})
            model.booster.load_model(str(path))
        else:
            model.booster = native.CatBoostRegressor()
            model.booster.load_model(str(path))
        model.vocabulary, model.rounds, model.metadata = raw["vocabulary"], raw["rounds"], raw["metadata"]
        model.native_parameters = raw["native_parameters"]
        return model
