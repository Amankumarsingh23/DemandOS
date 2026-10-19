"""Native LightGBM quantile objectives over the existing safe feature contract."""

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from demandos.boosting.models import GlobalModel
from demandos.boosting.config import BoostConfig
from demandos.features.config import FeatureConfig
from .contracts import levels


class QuantileLightGBM(GlobalModel):
    def __init__(self, alpha, features, config=BoostConfig(), feature_config=FeatureConfig()):
        levels([alpha])
        super().__init__("lightgbm", features, config, feature_config)
        self.alpha = float(alpha)

    def _fit_native(self, X, y, Xv, yv):
        c = self.config
        params = {"objective": "quantile", "alpha": self.alpha, "metric": "quantile", "learning_rate": c.learning_rate,
                  "num_leaves": c.num_leaves, "max_depth": c.max_depth, "min_child_samples": c.min_child_samples,
                  "lambda_l1": c.reg_alpha, "lambda_l2": c.reg_lambda, "feature_fraction": c.colsample,
                  "bagging_fraction": c.subsample, "bagging_freq": 1, "num_threads": c.threads,
                  "seed": c.seed, "deterministic": True, "force_col_wise": True, "verbosity": 1}
        training = lgb.Dataset(X, label=y, categorical_feature=self.categories)
        validation = lgb.Dataset(Xv, label=yv, reference=training, categorical_feature=self.categories) if Xv is not None else None
        self.booster = lgb.train(params, training, num_boost_round=c.iterations,
            valid_sets=[validation] if validation is not None else None,
            callbacks=[lgb.early_stopping(c.patience, verbose=False)] if validation is not None else [])
        self.rounds = self.booster.best_iteration or self.booster.current_iteration()
        self.native_parameters = params

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.metadata.update(quantile_level=self.alpha, target="observed daily units sold",
                             validation_pinball=self.booster.best_score.get("valid_0", {}).get("quantile"))
        return self

    def save(self, directory):
        super().save(directory)
        path = Path(directory)/"metadata.json"
        payload = json.loads(path.read_text())
        payload["quantile_alpha"] = self.alpha
        path.write_text(json.dumps(payload, default=str, indent=2)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        raw = json.loads((directory/"metadata.json").read_text())
        model = cls(raw["quantile_alpha"], raw["features"], BoostConfig(**raw["config"]), FeatureConfig(**raw["feature_config"]))
        if (raw["model_file"] != "model.txt" or raw["native_parameters"]["objective"] != "quantile"
                or raw["native_parameters"]["alpha"] != model.alpha):
            raise ValueError("Quantile checkpoint objective/alpha mismatch")
        model.booster = lgb.Booster(model_file=str(directory/"model.txt"))
        model.vocabulary, model.rounds, model.metadata = raw["vocabulary"], raw["rounds"], raw["metadata"]
        model.native_parameters = raw["native_parameters"]
        cutoff, target = pd.Timestamp(model.metadata.get("allowed_cutoff")), pd.Timestamp(model.metadata.get("training_target_max"))
        if pd.isna(cutoff) or pd.isna(target) or target > cutoff:
            raise ValueError("Missing/future quantile fitting labels")
        return model
