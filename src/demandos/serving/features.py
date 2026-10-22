"""Same Season 8/16 transformations, with explicit assumed availability selection."""
from dataclasses import asdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype, is_bool_dtype

from demandos.features.config import FeatureConfig
from demandos.features.contracts import feature_dictionary
from demandos.features.encoding import CategoryMapping, CATEGORICAL
from demandos.features.validation import generate, validate_features
from .core import require, Blocked, digest, file_hash, frame_hash

KEY = ["store_id", "item_id"]
CASE = [*KEY, "origin_date", "horizon", "target_date"]


def contract(root, config, mapping):
    modules = ["features/sql.py", "features/contracts.py", "features/validation.py", "features/encoding.py",
               "boosting/models.py", "quantiles/tree.py", "quantiles/baseline.py", "serving/features.py"]
    return dict(version="existing-origin-features-v1", transformation="Seasons 8/16 generate + native GlobalModel.matrix",
                config=asdict(config), mapping=mapping.as_dict(), fields=feature_dictionary(config),
                units="units_sold: recorded units/day; prices unavailable without verified knowledge stamps",
                history_days=config.lookback + 1, horizons=list(range(1, 29)), output=["q10", "q50", "q90"],
                availability="ASSUMED replay: observed sales available at end of event date; static calendar known in advance",
                implementations={p: file_hash(Path(root)/"src/demandos"/p) for p in modules})


def eligible_history(snapshot, origin):
    required = {*KEY, "date", "units_sold", "available_at"}
    require(required.issubset(snapshot), "INPUT_SCHEMA", sorted(required-set(snapshot)))
    frame = snapshot.copy(); origin = pd.Timestamp(origin)
    for name in ["date", "available_at"]:
        require(pd.api.types.is_datetime64_any_dtype(frame[name]), "INPUT_TYPES", name)
        require(frame[name].notna().all() and frame[name].eq(frame[name].dt.normalize()).all(), "INPUT_DATES", name)
    require(frame[KEY].notna().all().all(), "INPUT_NULL_KEYS")
    require(not frame.duplicated(KEY+["date"]).any(), "DUPLICATE_SOURCE_KEYS")
    require(is_integer_dtype(frame.units_sold), "INPUT_TYPES", "sales must be integer units")
    require(frame.units_sold.notna().all() and frame.units_sold.ge(0).all(), "INVALID_SALES")
    require(frame.available_at.ge(frame.date).all(), "AVAILABILITY_BEFORE_EVENT")
    return frame.loc[frame.date.le(origin)&frame.available_at.le(origin)].copy()


def extract(snapshot, calendar, metadata, origin, config, mapping):
    origin = pd.Timestamp(origin)
    require(not pd.isna(origin) and origin == origin.normalize(), "INVALID_ORIGIN")
    history = eligible_history(snapshot, origin)
    require(not metadata.empty and not metadata.duplicated(KEY).any(), "COHORT_KEYS")
    require(set(map(tuple, history[KEY].drop_duplicates().to_numpy())) == set(map(tuple, metadata[KEY].to_numpy())), "SOURCE_COHORT_COVERAGE")
    for _, g in history.groupby(KEY):
        require(g.date.max() == origin, "STALE_SOURCE", "as-of origin day missing")
        recent = g[g.date.ge(origin-pd.Timedelta(days=config.lookback))]
        require(len(recent) == config.lookback+1, "INCOMPLETE_HISTORY", "late/missing sales are not zero-filled")
    with duckdb.connect() as con:
        con.execute("set threads=2")
        con.register("feature_sales", history[[*KEY, "date", "units_sold"]])
        con.register("feature_keys", metadata[[*KEY, "state_id", "dept_id", "cat_id", "first_observed_date"]])
        con.register("feature_calendar", calendar)
        try:
            frame = generate(con, [origin], config, mapping).features
        except (ValueError, AssertionError) as exc:
            raise Blocked("FEATURE_CONTRACT", str(exc)) from exc
    validate_frame(frame, config, mapping)
    dates = pd.date_range(origin+pd.Timedelta(days=1), periods=28)
    frozen_calendar = calendar[calendar.date.isin(dates)].sort_values("date").reset_index(drop=True)
    source_digest = digest(dict(history=frame_hash(history[[*KEY, "date", "units_sold", "available_at"]], [*KEY, "date"]),
                                calendar=frame_hash(frozen_calendar), metadata=frame_hash(metadata, KEY)))
    return frame, source_digest


def validate_frame(frame, config, mapping):
    try:
        validate_features(frame, config, tuple(range(1, 29)))
    except (ValueError, TypeError, KeyError) as exc:
        raise Blocked("FEATURE_CONTRACT", exc) from exc
    for field in feature_dictionary(config):
        x = frame[field["name"]]; kind = field["type"]
        valid = (is_integer_dtype(x) if kind == "int64" else is_float_dtype(x) and x.dtype.itemsize==8 if kind == "float64" else
                 is_bool_dtype(x) if kind == "bool" else pd.api.types.is_datetime64_any_dtype(x) if kind == "date32" else
                 pd.api.types.is_string_dtype(x))
        require(valid, "FEATURE_TYPES", field["name"])
        if kind=="date32":
            require(x.dropna().eq(x.dropna().dt.normalize()).all(),"FEATURE_DAILY_TIMESTAMPS",field["name"])
    for name in ["first_observed_date","bounded_history_start"]:
        require(frame[name].le(frame.origin_date).all(),"FUTURE_HISTORY_PROVENANCE",name)
    for name in CATEGORICAL:
        expected = frame[name].astype(str).map(mapping.vocabulary[name]).fillna(0).astype(int)
        require(np.array_equal(expected, frame[name+"_code"].to_numpy(int)), "ENCODING_MISMATCH", name)


def parity(reference, serving, *, atol=1e-10):
    require(set(reference)==set(serving), "FEATURE_PARITY_SCHEMA")
    a = reference.sort_values(CASE).reset_index(drop=True)
    b = serving.sort_values(CASE).reset_index(drop=True)[a.columns]
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, atol=atol, rtol=0)
    except AssertionError as exc:
        raise Blocked("TRAINING_SERVING_SKEW", str(exc)[:500]) from exc
    return dict(rows=len(b), columns=len(b.columns), tolerance=atol, passed=True)
