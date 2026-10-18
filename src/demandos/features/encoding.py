"""Stable training-only ID vocabularies. Code zero means unknown, never ordinal rank."""

from dataclasses import dataclass

import pandas as pd


CATEGORICAL = ("store_id", "state_id", "item_id", "dept_id", "cat_id")


@dataclass(frozen=True)
class CategoryMapping:
    fitted_through: str
    vocabulary: dict

    def transform(self, frame):
        result = frame.copy()
        cutoff = pd.Timestamp(self.fitted_through)
        if result.origin_date.isna().any() or not pd.to_datetime(result.origin_date).ge(cutoff).all():
            raise ValueError("Category mapping was fitted after a forecast origin")
        for column in CATEGORICAL:
            if column not in result or result[column].isna().any():
                raise ValueError("Canonical identifiers must be present")
            result[f"{column}_code"] = result[column].astype(str).map(self.vocabulary[column]).fillna(0).astype("int64")
        result["mapping_fit_end"] = cutoff
        return result

    def as_dict(self):
        return {"fitted_through": self.fitted_through, "unknown_code": 0, "vocabulary": self.vocabulary,
                "fit_policy": "Identifiers observed on positive-sales training dates only; no target encoding"}


def fit_categories(metadata, fit_end):
    required = [*CATEGORICAL, "date"]
    if not set(required).issubset(metadata):
        raise ValueError("Dated identifier observations required")
    cutoff = pd.Timestamp(fit_end)
    if pd.isna(cutoff) or cutoff != cutoff.normalize():
        raise ValueError("Valid category-fit date required")
    # Filter before examining categories: even invalid future values cannot affect fit.
    prefix = metadata.loc[pd.to_datetime(metadata.date).le(cutoff), required]
    if prefix.empty or prefix.isna().any().any():
        raise ValueError("No complete training-only category observations")
    vocabulary = {column: {value: index + 1 for index, value in enumerate(sorted(prefix[column].astype(str).unique()))}
                  for column in CATEGORICAL}
    return CategoryMapping(str(cutoff.date()), vocabulary)
