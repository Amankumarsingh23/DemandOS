"""One declared type, grain, role, source, availability and null policy per column."""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .encoding import CATEGORICAL


KEYS = ["store_id", "item_id", "origin_date", "horizon"]
TYPES = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_(), "date32": pa.date32()}


def origin_dictionary(config):
    entries = []
    def add(name, dtype, role, source, availability, null_policy):
        entries.append({"name": name, "type": dtype, "role": role, "grain": "store-item-origin",
                        "source": source, "availability": availability, "null_policy": null_policy})
    for column in CATEGORICAL:
        add(column, "string", "feature_key" if column in ("store_id", "item_id") else "feature", "validated static store/product dimension",
            "Assumed immutable identifier hierarchy known at origin; not learned from targets", "Forbidden")
    add("origin_date", "date32", "key", "requested origin schedule", "End of day after origin sales are available", "Forbidden")
    for name in ("first_observed_date", "bounded_history_start", "max_sales_source_date"):
        add(name, "date32", "audit", "observed daily sales history", "On/before origin", "Forbidden")
    for name in ("history_observations", "history_distinct_dates", "history_invalid_rows"):
        add(name, "int64", "audit", "bounded daily history validation", "On/before origin", "Forbidden")
    add("sales_at_origin", "int64", "feature", "observed y_origin", "End of origin day", "Forbidden")
    for lag in config.lags:
        add(f"origin_lag_{lag}", "int64", "feature", f"observed y_(origin-{lag})", "At source day end, <= origin", "Null if source predates observed history")
        add(f"origin_lag_{lag}_source_date", "date32", "audit", f"exact origin minus {lag} calendar days", "<= origin", "Null exactly when lag unavailable")
    for window in config.windows:
        for statistic in ("mean", "sum", "std", "max", "zero_fraction", "positive_count"):
            dtype = "int64" if statistic in ("sum", "max", "positive_count") else "float64"
            add(f"rolling_{statistic}_{window}_at_origin", dtype, "feature", f"last {window} daily observations, origin inclusive; std uses population ddof=0",
                "On/before origin", f"Null unless {window} complete daily observations exist; zero observations stay zero")
        add(f"rolling_observations_{window}", "int64", "feature", f"count of actually observed days in last {window}", "On/before origin", "Never null; incomplete count is not zero-filled sales")
    for name in ("mean_previous_7_at_origin", "level_change_7", "relative_level_change_7"):
        add(name, "float64", "feature", "previous seven days (origin-13..origin-7); recent minus previous mean; relative difference divided by previous mean",
            "On/before origin", "Null without 14 daily observations; relative value also null for zero prior mean")
    for name in ("last_known_price", "prior_known_price", "historical_price_change", "historical_relative_price_4"):
        add(name, "float64", "feature", "validated weekly price joined to explicitly verified observed_at/available_at evidence",
            "Both observed_at and available_at must be <= origin; unavailable by default",
            "Null without evidence; change requires positive prior price; relative price requires four known weeks with positive mean")
    for name in ("last_price_observed_at", "last_price_available_at", "prior_price_observed_at", "prior_price_available_at"):
        add(name, "date32", "audit", "explicit historical-price evidence", "<= origin", "Null if corresponding historical price unavailable")
    add("known_price_weeks", "int64", "feature", "number of distinct verified historical weeks", "On/before origin", "Zero without evidence")
    for name in ("last_price_missing", "prior_price_missing"):
        add(name, "bool", "feature", "historical price availability", "On/before origin", "Never null; true means unavailable, not free")
    return entries


def feature_dictionary(config):
    entries = origin_dictionary(config)
    def add(name, dtype, role, source, availability, null_policy="Forbidden"):
        entries.append({"name": name, "type": dtype, "role": role, "grain": "store-item-origin-horizon",
                        "source": source, "availability": availability, "null_policy": null_policy})
    add("horizon", "int64", "feature_key", "requested 1..28 day direct horizon", "At forecast issuance")
    add("target_date", "date32", "target_index", "origin + horizon calendar days", "Known at origin")
    for name in ("target_weekday", "target_month", "target_event_count", "target_snap"):
        add(name, "int64", "feature", "validated calendar; weekday Monday=0; SNAP chosen from store state", "Assumed publicly scheduled/known at origin")
    for name in ("target_weekend", "target_has_event", "target_event_cultural", "target_event_national", "target_event_religious", "target_event_sporting"):
        add(name, "bool", "feature", "validated target calendar dates and both event slots", "Assumed publicly scheduled/known at origin", "Never null; missing event names mean no listed event")
    for name in ("target_weekday_sin", "target_weekday_cos", "target_month_sin", "target_month_cos"):
        add(name, "float64", "feature", "deterministic cyclic calendar encoding", "Known at origin")
    for column in CATEGORICAL:
        add(f"{column}_code", "int64", "feature", "saved training-only category vocabulary", "Fit cutoff must not exceed any origin", "Code 0 means unknown; no ordinal meaning")
    add("mapping_fit_end", "date32", "audit", "saved preprocessing provenance", "Before all generated origins")
    return entries


def label_dictionary():
    return [{"name": name, "type": "string" if name in ("store_id", "item_id") else "int64" if name in ("horizon", "target_units_sold") else "date32",
             "role": "label" if name == "target_units_sold" else "label_key", "grain": "store-item-origin-horizon",
             "source": "actual observed sales at target_date" if name == "target_units_sold" else "feature keys",
             "availability": "Offline only, after target day; must be <= supervised label cutoff", "null_policy": "Forbidden"}
            for name in [*KEYS, "target_date", "target_units_sold"]]


def predictive_columns(config):
    return [entry["name"] for entry in feature_dictionary(config) if entry["role"] in ("feature", "feature_key")]


def cast_frame(frame, dictionary):
    names = [entry["name"] for entry in dictionary]
    if set(frame) != set(names):
        raise ValueError(f"Unexpected/missing contract columns: {sorted(set(frame) ^ set(names))}")
    result = frame[names].copy()
    for entry in dictionary:
        name, dtype = entry["name"], entry["type"]
        if dtype == "date32":
            result[name] = pd.to_datetime(result[name])
        else:
            result[name] = result[name].astype({"string": "string", "int64": "Int64", "float64": "float64", "bool": "boolean"}[dtype])
    return result


def write_parquet(frame, path, dictionary):
    frame = cast_frame(frame, dictionary)
    schema = pa.schema([pa.field(entry["name"], TYPES[entry["type"]]) for entry in dictionary])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False), path, compression="zstd")
