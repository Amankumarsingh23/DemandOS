"""Small validation helpers shared by source readers and joins."""

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Source data violates the ingestion contract."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def columns(frame, expected, source):
    missing = set(expected) - set(frame.columns)
    require(not missing, f"{source}: missing columns {sorted(missing)}")


def nonnull(frame, names, source):
    require(not frame[list(names)].isna().any().any(), f"{source}: null required values")
    for name in names:
        if pd.api.types.is_string_dtype(frame[name]):
            require(not frame[name].str.strip().eq("").any(), f"{source}: blank {name}")


def unique(frame, keys, source):
    require(not frame.duplicated(keys).any(), f"{source}: duplicate key {keys}")


def integer_values(values, source, minimum=0):
    numbers = pd.to_numeric(values, errors="coerce")
    array = np.asarray(numbers, dtype="float64")
    require(
        bool(np.isfinite(array).all() and (array >= minimum).all()
             and (array == np.floor(array)).all() and (array <= 2**31 - 1).all()),
        f"{source}: expected nonnegative finite int32 values (minimum {minimum})",
    )
    return numbers.astype("int32")
