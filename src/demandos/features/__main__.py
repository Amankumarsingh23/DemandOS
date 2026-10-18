"""Generate origin-safe feature matrices, without fitting forecasting models."""

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inference-only", action="store_true", help="Do not generate supervised training rows/labels")
    parser.add_argument("--cohort", type=Path, help="Explicit CSV/Parquet with store_id,item_id only")
    parser.add_argument("--price-evidence", type=Path, help="Historical-price keys with explicit observed_at,available_at")
    args = parser.parse_args()
    summary, target = run(args.root, load_config(args.config or args.root / "configs/features.toml"),
        inference_only=args.inference_only, cohort_path=args.cohort, price_evidence_path=args.price_evidence)
    print(f"Features passed: {summary['cohort_series']} cohort series; {summary['evaluation']['rows']} inference rows; "
          f"{summary['training_labels']} training labels; {summary['runtime_seconds']:.2f}s; {target}")


if __name__ == "__main__":
    main()
