import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Season 5 diagnostics and forecast-origin contracts; no model training")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        from .pipeline import run, load_config
        summary, output = run(args.root, load_config(args.config or args.root / "configs/timeseries.toml"))
        print(f"PASS: {len(summary['cohort'])} real diagnostic series; {summary['runtime_seconds']:.2f}s")
        print(f"Outputs: {output}")
        return 0
    except (ValueError, FileNotFoundError, ImportError) as error:
        print(f"Diagnostics stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
