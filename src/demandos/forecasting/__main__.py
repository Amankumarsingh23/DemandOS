"""Run the fixed-origin observed-sales experiment."""

import argparse
from pathlib import Path

from .pipeline import load_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config or args.root / "configs/forecasting.toml")
    summary, target = run(args.root, config)
    print(f"Forecast experiment: {summary['fit_counts']}; {summary['runtime_seconds']:.2f}s; {target}")


if __name__ == "__main__":
    main()
