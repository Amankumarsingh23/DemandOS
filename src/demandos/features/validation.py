"""Feature availability guards and physically separate supervised-label joins."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from demandos.eda.queries import fetch_frame
from demandos.timeseries.temporal import FeatureEvidence, validate_feature
from .contracts import KEYS, feature_dictionary, predictive_columns, cast_frame, label_dictionary
from .sql import origin_features, expand_calendar


def validate_features(frame, config, horizons):
    expected = {entry["name"] for entry in feature_dictionary(config)}
    if set(frame) != expected:
        raise ValueError(f"Undeclared predictive/audit fields: {sorted(set(frame) ^ expected)}")
    if frame.empty or frame[KEYS + ["target_date"]].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Empty, missing or duplicate feature keys")
    if not frame.target_date.eq(frame.origin_date + pd.to_timedelta(frame.horizon.astype(int), unit="D")).all():
        raise ValueError("Target must equal origin plus horizon")
    for _, group in frame.groupby(["store_id", "item_id", "origin_date"]):
        if sorted(group.horizon) != sorted(horizons):
            raise ValueError("Incomplete or duplicated direct horizon set")
    if not frame.max_sales_source_date.eq(frame.origin_date).all() or not frame.mapping_fit_end.le(frame.origin_date).all():
        raise ValueError("Future historical observations or fitted preprocessing")
    for lag in config.lags:
        value, source = frame[f"origin_lag_{lag}"], frame[f"origin_lag_{lag}_source_date"]
        if not value.isna().eq(source.isna()).all() or not source[source.notna()].eq(
            frame.loc[source.notna(), "origin_date"] - pd.Timedelta(days=lag)).all():
            raise ValueError("Origin-relative lag timestamp mismatch")
    for prefix in ("last", "prior"):
        price = frame[f"{prefix}_known_price"]
        observed, available = frame[f"{prefix}_price_observed_at"], frame[f"{prefix}_price_available_at"]
        valid = price.notna()
        if not valid.eq(observed.notna() & available.notna()).all():
            raise ValueError("Historical price lacks as-of provenance")
        if not observed[valid].le(frame.loc[valid, "origin_date"]).all() or not available[valid].le(frame.loc[valid, "origin_date"]).all():
            raise ValueError("Future historical price information")
        if not frame[f"{prefix}_price_missing"].eq(~valid).all():
            raise ValueError("Price missingness flag disagrees with price availability")
    # Reuse Season 5's evidence checks once per origin (row-level dates above are exact).
    for origin, group in frame.groupby("origin_date"):
        validate_feature(FeatureEvidence("sales_at_origin", None, group.max_sales_source_date.max(), origin), origin)
        validate_feature(FeatureEvidence("target_calendar", None, group.target_date.max(), origin, kind="calendar"), origin)
        for prefix in ("last", "prior"):
            known = group[group[f"{prefix}_known_price"].notna()]
            if not known.empty:
                validate_feature(FeatureEvidence(f"{prefix}_historical_price", None, known[f"{prefix}_price_observed_at"].max(),
                    known[f"{prefix}_price_available_at"].max(), kind="price", availability_verified=True), origin)
    for entry in feature_dictionary(config):
        column = frame[entry["name"]]
        if entry["type"] in ("int64", "float64") and not np.isfinite(column.dropna().to_numpy(float)).all():
            raise ValueError(f"Nonfinite feature: {entry['name']}")
        if entry["null_policy"].startswith("Forbidden") and column.isna().any():
            raise ValueError(f"Missing required feature: {entry['name']}")
    return {"rows": len(frame), "series": len(frame[["store_id", "item_id"]].drop_duplicates()),
            "origins": frame.origin_date.nunique(), "unique_keys": True, "target_dates_match": True,
            "availability_passed": True, "null_counts": {name: int(value) for name, value in frame.isna().sum().items()}}


def supervised_labels(con, features, label_cutoff):
    cutoff = pd.Timestamp(label_cutoff)
    if pd.isna(cutoff) or features.target_date.gt(cutoff).any():
        raise ValueError("Supervised labels extend into excluded evaluation targets or past label cutoff")
    con.register("feature_label_keys", features[KEYS + ["target_date"]])
    labels = fetch_frame(con, """select k.*, f.units_sold as target_units_sold
        from feature_label_keys k left join feature_sales f
        on f.store_id=k.store_id and f.item_id=k.item_id and f.date=k.target_date
        order by k.origin_date,k.store_id,k.item_id,k.horizon""")
    if len(labels) != len(features) or labels.duplicated(KEYS).any() or labels.target_units_sold.isna().any():
        raise ValueError("Supervised label join missing or multiplying rows")
    units = labels.target_units_sold.to_numpy(float)
    if not np.isfinite(units).all() or (units < 0).any() or (units != np.floor(units)).any():
        raise ValueError("Invalid supervised observed-sales labels")
    return cast_frame(labels, label_dictionary())


@dataclass
class FeatureBatch:
    features: pd.DataFrame
    labels: pd.DataFrame | None
    predictive_columns: list
    validation: dict


def generate(con, origins, config, mapping, *, horizons=tuple(range(1, 29)), inference_only=True,
             label_cutoff=None, verified_prices=None):
    """Return features and labels separately; inference never queries target sales."""
    index = pd.DatetimeIndex(pd.to_datetime(origins))
    if index.empty or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Unique chronological origins required")
    historical = pd.concat([origin_features(con, origin, config, verified_prices) for origin in index], ignore_index=True)
    frame = cast_frame(mapping.transform(expand_calendar(con, historical, horizons)), feature_dictionary(config))
    validation = validate_features(frame, config, horizons)
    labels = None
    if not inference_only:
        if label_cutoff is None:
            raise ValueError("Offline labels require an explicit permitted label cutoff")
        labels = supervised_labels(con, frame, label_cutoff)
    return FeatureBatch(frame, labels, predictive_columns(config), validation)
