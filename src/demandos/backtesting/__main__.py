"""Run reproducible rolling-origin evaluation; no new model implementations."""

import argparse
from pathlib import Path

from .design import load_config
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary, target = run(args.root, load_config(args.config or args.root / "configs/backtesting.toml"))
    print(f"Backtest {summary['fit_counts']}; {summary['runtime_seconds']:.2f}s; {target}")


if __name__ == "__main__":
    main()
