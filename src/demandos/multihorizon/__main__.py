"""Run the bounded real-data multi-horizon architecture comparison."""

import argparse
from pathlib import Path
from .config import load_config
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary, output = run(args.root, load_config(args.config or args.root/"configs/multihorizon.toml"))
    print(f"Strategies passed: {summary['native_models']} native models; "
          f"{summary['table_rows']['forecasts']:,} forecast rows; {summary['runtime_seconds']:.2f}s; {output}")


if __name__ == "__main__":
    main()
