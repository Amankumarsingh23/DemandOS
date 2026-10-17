"""Command-line entry point; never downloads or manufactures source data."""

import argparse
from pathlib import Path
import sys

import pandas as pd

from .pipeline import run_pipeline
from .validation import ValidationError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate and ingest local M5 CSVs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--chunk-size", type=int, default=128, help="Wide sales rows per chunk")
    args = parser.parse_args(argv)
    try:
        report = run_pipeline(args.data_dir, args.chunk_size)
    except (FileNotFoundError, ValidationError, pd.errors.ParserError, pd.errors.EmptyDataError, pd.errors.MergeError) as error:
        print(f"Ingestion stopped: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {report['daily_rows']:,} rows; report: {args.data_dir / 'processed/quality_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
