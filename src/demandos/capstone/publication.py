"""Single local writer, immutable complete bundle, same-filesystem atomic rename."""
import os
from pathlib import Path
import shutil
import uuid
import pandas as pd

from demandos.serving.core import atomic_json, file_hash, read_json, require, writer_lock
from .config import run_identity


def seal(directory):
    directory = Path(directory)
    files = {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "completed.json"}
    atomic_json(directory / "completed.json", dict(files=files))


def check_seal(directory):
    directory = Path(directory)
    record = read_json(directory / "completed.json")
    actual = {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
              if p.is_file() and p.name != "completed.json"}
    require(actual == record["files"], "BUNDLE_DIGEST_MISMATCH")
    manifest = read_json(directory / "manifest.json")
    return run_identity(manifest)


def equivalent(first, second, *, tolerance=1e-8):
    require(read_json(first / "manifest.json") == read_json(second / "manifest.json"), "RUN_ID_CONFLICT")
    one = {p.relative_to(first) for p in first.rglob("*.parquet")}
    require(one == {p.relative_to(second) for p in second.rglob("*.parquet")}, "RUN_TABLE_CONFLICT")
    for relative in one:
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(first / relative), pd.read_parquet(second / relative),
                                          check_exact=False, atol=tolerance, rtol=0)
        except AssertionError as exc:
            raise ValueError(f"RUN_CONTENT_CONFLICT: {relative}") from exc
    for name in ["evidence.json", "faults.json", "summary.json"]:
        require(read_json(first / name) == read_json(second / name), "RUN_CONTENT_CONFLICT", name)


def publish(base, build, audit):
    base = Path(base)
    with writer_lock(base):
        stage = base / (".s-" + uuid.uuid4().hex[:12])
        stage.mkdir()
        try:
            build(stage)
            manifest = read_json(stage / "manifest.json")
            identifier = run_identity(manifest)
            audit(stage, sealed=False)
            seal(stage)
            check_seal(stage)
            target = base / "runs" / identifier[:20]
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                require(check_seal(target) == identifier, "RUN_ID_CONFLICT")
                audit(target)
                equivalent(target, stage)
                reused = True
            else:
                os.rename(stage, target)
                reused = False
            atomic_json(base / "latest_verified.json", dict(run_id=identifier, path=target.relative_to(base).as_posix()))
            return target, reused
        finally:
            # This path is generated beneath the explicit base, never user input.
            if stage.exists():
                require(stage.resolve().is_relative_to(base.resolve()), "UNSAFE_STAGING_CLEANUP")
                shutil.rmtree(stage)
