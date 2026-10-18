"""Direct models and an explicitly independent, complete-target vector wrapper."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from demandos.boosting.models import GlobalModel, TARGET, validate_rows
from demandos.boosting.dataset import row_digest
from demandos.features.contracts import KEYS
from .vector import HORIZONS, ORIGIN_KEYS, VectorLayout, attach_vectors


def fit_model(training, features, config, feature_config, cutoff, directory, records,
              architecture, phase, horizon=0, validation=None, validation_end=None, probe=None):
    """Record every native fit and abort on failure; no architecture substitution."""
    fit_id = f"{architecture}/{phase}/{pd.Timestamp(cutoff):%Y%m%d}/h{horizon:02}"
    record = {"fit_id": fit_id, "architecture": architecture, "phase": phase, "horizon": horizon,
              "cutoff": pd.Timestamp(cutoff), "rows": len(training),
              "training_origins": training.origin_date.nunique(),
              "training_series": len(training[["store_id", "item_id"]].drop_duplicates()),
              "target_max": training.target_date.max(), "fit_digest": row_digest(training)}
    try:
        model = GlobalModel("lightgbm", features, config, feature_config)
        model.fit(training, allowed_cutoff=cutoff, validation=validation, validation_cutoff=validation_end)
        model.save(directory)
        probe = validation.drop(columns=TARGET) if validation is not None else probe
        if probe is None:
            raise ValueError("Origin-valid inference probe required for save/load verification")
        probe = probe.iloc[:96]
        restored = GlobalModel.load(directory)
        start = perf_counter()
        predicted = model.predict(probe)
        prediction_seconds = perf_counter() - start
        np.testing.assert_allclose(predicted.raw_prediction, restored.predict(probe).raw_prediction, rtol=1e-10, atol=1e-10)
        record.update(status="ok", reason=None, rounds=model.rounds,
                      training_seconds=model.metadata["fit_seconds"], probe_prediction_seconds=prediction_seconds,
                      model_bytes=(Path(directory)/"model.txt").stat().st_size,
                      metadata_bytes=(Path(directory)/"metadata.json").stat().st_size,
                      warning_count=len(model.metadata["warnings"])+len(model.metadata["native_warning_lines"]),
                      save_load_equal=True)
        records.append(record)
        return model
    except Exception as error:
        records.append({**record, "status": "failed", "reason": f"{type(error).__name__}: {error}"})
        raise RuntimeError(f"Required native fit failed ({fit_id}); publication blocked: {error}") from error


class HorizonModels:
    """28 global models; horizon h consumes that horizon's origin-safe feature row."""
    def __init__(self, models):
        if set(models) != set(HORIZONS):
            raise ValueError("Exactly 28 independently fitted horizon models required")
        self.models = models

    @classmethod
    def fit(cls, training, features, config, feature_config, cutoff, directory, records,
            *, architecture="direct", phase="evaluation", rounds=None, validation=None, validation_end=None, probe=None):
        validate_rows(training, labelled=True, cutoff=cutoff)
        models = {}
        for h in HORIZONS:
            selected = training[training.horizon.eq(h)].copy()
            valid = validation[validation.horizon.eq(h)].copy() if validation is not None else None
            current = replace(config, iterations=int(rounds[h])) if rounds is not None else config
            models[h] = fit_model(selected, features, current, feature_config, cutoff, Path(directory)/f"h{h:02}",
                                  records, architecture, phase, h, valid, validation_end,
                                  probe[probe.horizon.eq(h)] if probe is not None else None)
        return cls(models)

    def predict(self, inference):
        validate_rows(inference)
        if inference.groupby(ORIGIN_KEYS).horizon.size().ne(28).any():
            raise ValueError("Complete 28-horizon inference required")
        return pd.concat([self.models[h].predict(inference[inference.horizon.eq(h)]) for h in HORIZONS],
                         ignore_index=True).sort_values(KEYS).reset_index(drop=True)

    @property
    def rounds(self):
        return {h: model.rounds for h, model in self.models.items()}

    @classmethod
    def load(cls, directory):
        return cls({h: GlobalModel.load(Path(directory)/f"h{h:02}") for h in HORIZONS})


class IndependentMultiOutput:
    """Vector API with 28 independent LightGBM objectives; no joint output loss."""
    def __init__(self, layout, estimators):
        self.layout, self.estimators = layout, estimators

    @classmethod
    def fit(cls, X, Y, layout, features, config, feature_config, cutoff, directory, records,
            *, phase="evaluation", rounds=None, validation=None, validation_end=None, probe=None):
        training = attach_vectors(layout, X, Y, cutoff)
        valid = attach_vectors(layout, *validation, validation_end) if validation is not None else None
        estimators = HorizonModels.fit(training, features, config, feature_config, cutoff, directory, records,
                                      architecture="independent_multioutput", phase=phase, rounds=rounds,
                                      validation=valid, validation_end=validation_end,
                                      probe=layout.unpack(probe) if probe is not None else None)
        wrapper = cls(layout, estimators)
        wrapper.save(directory)
        return wrapper

    def predict_vector(self, X):
        """Return raw/scored arrays of shape (number of origin rows, 28)."""
        long = self.layout.unpack(X)
        predicted = self.estimators.predict(long)
        keys = X[ORIGIN_KEYS].reset_index(drop=True)
        raw, scored = [], []
        for h in HORIZONS:
            selected = predicted[predicted.horizon.eq(h)]
            aligned = keys.merge(selected[ORIGIN_KEYS+["raw_prediction", "prediction"]], on=ORIGIN_KEYS, validate="one_to_one")
            raw.append(aligned.raw_prediction.to_numpy(float))
            scored.append(aligned.prediction.to_numpy(float))
        return np.column_stack(raw), np.column_stack(scored)

    def predict(self, X):
        raw, scored = self.predict_vector(X)
        blocks = []
        for h in HORIZONS:
            block = X[ORIGIN_KEYS].reset_index(drop=True).copy()
            block["horizon"] = h
            block["target_date"] = pd.to_datetime(block.origin_date) + pd.Timedelta(days=h)
            block["model"] = "lightgbm"
            block["raw_prediction"], block["prediction"] = raw[:,h-1], scored[:,h-1]
            blocks.append(block)
        return pd.concat(blocks, ignore_index=True).sort_values(KEYS).reset_index(drop=True)

    def save(self, directory):
        # Native children are saved during fit; persist the vector input contract.
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for h, model in self.estimators.models.items():
            model.save(directory/f"h{h:02}")
        payload = {"kind": "independent_multioutput", "native_models": 28,
                   "joint_objective": False, "layout": asdict(self.layout), "partial_targets": False}
        (directory/"wrapper.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        raw = json.loads((Path(directory)/"wrapper.json").read_text())
        if raw["kind"] != "independent_multioutput" or raw["native_models"] != 28 or raw["joint_objective"] or raw["partial_targets"]:
            raise ValueError("Invalid independent vector wrapper state")
        return cls(VectorLayout(**raw["layout"]), HorizonModels.load(directory))
