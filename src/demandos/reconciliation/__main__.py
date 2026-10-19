"""Run bounded Season 15 reconciliation against the existing verified M5 warehouse."""

import argparse
from pathlib import Path
from .pipeline import run, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary, output = run(args.root, load_config(args.config or args.root/"configs/reconciliation.toml"))
    print(f"Reconciliation passed: {summary['cohort_bottom']} bottom series; {summary['common_cases_per_method']:,} cases/method; "
          f"{summary['runtime_seconds']:.1f}s; {output}", flush=True)


if __name__ == "__main__":
    main()
