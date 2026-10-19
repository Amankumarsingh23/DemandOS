"""Train real M5 quantile forecasts, evaluate and verify temporal isolation."""

import argparse
from pathlib import Path
from .pipeline import run, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary, output = run(args.root, load_config(args.config or args.root/"configs/quantiles.toml"))
    print(f"Quantile verification passed: {summary['table_rows']['forecasts']:,} forecasts; "
          f"{summary['runtime_seconds']:.2f}s; {output}", flush=True)


if __name__ == "__main__":
    main()
