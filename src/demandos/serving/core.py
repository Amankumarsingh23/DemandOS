"""Canonical identities, explicit hard failures and local atomic file operations."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

STATUS = "LOCAL_HISTORICAL_REPLAY; SIMULATED_DECISIONS; NO_LIVE_DISPATCH"


class Blocked(ValueError):
    def __init__(self, signal, evidence=""):
        self.signal, self.evidence = signal, str(evidence)
        super().__init__(f"{signal}: {evidence}")


def require(condition, signal, evidence=""):
    if not condition:
        raise Blocked(signal, evidence)


def canonical(value):
    def convert(x):
        if isinstance(x, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(x).isoformat()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return x.as_posix()
        raise TypeError(type(x).__name__)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=convert).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_hash(frame, keys=None):
    value = frame.sort_values(keys).reset_index(drop=True) if keys else frame.reset_index(drop=True)
    # Canonical logical content; no Parquet writer timestamps or pandas indices.
    return digest(json.loads(value.to_json(orient="split", date_format="iso", double_precision=15, index=False)))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(".tmp-" + uuid.uuid4().hex[:16])
    try:
        with staging.open("xb") as handle:
            handle.write(canonical(value) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


@contextmanager
def writer_lock(directory, name="writer.lock"):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise Blocked("LOCAL_WRITER_BUSY", path) from exc
    try:
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        yield
    finally:
        path.unlink()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
