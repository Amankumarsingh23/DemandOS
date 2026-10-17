"""Run with python -m demandos.eda --root <repository>."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce Season 4 EDA from the verified local warehouse")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, help="Defaults to <root>/configs/eda.toml")
    parser.add_argument("--start", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive YYYY-MM-DD; use training cutoff to avoid future labels")
    args = parser.parse_args(argv)
    try:
        from .pipeline import run

        config = load_config(args.config or args.root / "configs/eda.toml")
        if args.start:
            config = replace(config, start_date=args.start)
        if args.end:
            config = replace(config, end_date=args.end)
        report, output = run(args.root, config)
        print(f"EDA PASS: {report['baseline']['daily_rows']:,} observations; {report['runtime_seconds']:.2f}s")
        print(f"Outputs: {output}")
        return 0
    except (FileNotFoundError, ValueError, ImportError) as error:
        print(f"EDA stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
