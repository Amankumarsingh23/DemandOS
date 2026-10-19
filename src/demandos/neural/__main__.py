"""Train and verify bounded global LSTM/GRU forecasts on the local M5 warehouse."""

import argparse
from pathlib import Path
from .config import load_config
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary, output = run(args.root, load_config(args.config or args.root/"configs/neural.toml"))
    print(f"Neural verification passed: {summary['table_rows']['forecasts']:,} rows, "
          f"{summary['runtime_seconds']:.1f}s; {output}", flush=True)


if __name__ == "__main__":
    main()
