"""A frozen validation-selected convex combination, with exact prediction keys."""

import numpy as np
import pandas as pd

from demandos.features.contracts import KEYS
from demandos.boosting.models import TARGET


def blend(pooled, seasonal, weight):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Convex weight required")
    columns = KEYS + ["target_date"]
    for frame in (pooled, seasonal):
        if frame.duplicated(columns).any() or not np.isfinite(frame[["raw_prediction", "prediction"]].to_numpy(float)).all():
            raise ValueError("Finite unique aligned component predictions required")
    joined = pooled.merge(seasonal, on=columns, how="outer", suffixes=("_pooled", "_seasonal"), indicator=True, validate="one_to_one")
    if joined._merge.ne("both").any():
        raise ValueError("Ensemble components must have exactly identical keys")
    output = joined[columns].copy()
    # Combine each component's scored forecast; do not feed negative raw values back.
    output["raw_prediction"] = weight*joined.prediction_pooled+(1-weight)*joined.prediction_seasonal
    output["prediction"] = np.maximum(output.raw_prediction, 0)
    output["model"] = "ensemble"
    return output.sort_values(KEYS).reset_index(drop=True)


def select_weight(pooled, seasonal, validation_labels, weights, *, validation_end, first_evaluation):
    end = pd.Timestamp(validation_end)
    if end >= pd.Timestamp(first_evaluation) or pd.to_datetime(validation_labels.target_date).gt(end).any():
        raise ValueError("Ensemble tuning must precede all evaluation targets")
    records = []
    for weight in weights:
        predicted = blend(pooled, seasonal, weight)
        joined = predicted.merge(validation_labels[KEYS+["target_date", TARGET]], on=KEYS+["target_date"], how="outer", indicator=True, validate="one_to_one")
        if joined._merge.ne("both").any():
            raise ValueError("Validation predictions and labels must match exactly")
        records.append({"pooled_weight": weight, "seasonal_weight": 1-weight,
                        "mae": float(np.mean(np.abs(joined[TARGET]-joined.prediction))),
                        "rows": len(joined), "target_max": joined.target_date.max()})
    table = pd.DataFrame(records).sort_values(["mae", "pooled_weight"], kind="stable")
    return float(table.iloc[0].pooled_weight), table.reset_index(drop=True)
