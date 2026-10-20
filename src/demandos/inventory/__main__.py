"""Run bounded scenario-based inventory decisions using verified M5 artifacts."""
import argparse
from pathlib import Path
from .pipeline import run, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary, target = run(args.root, load_config(args.config or args.root/"configs/inventory.toml"))
    print(f"SIMULATED inventory verification passed: {summary['table_rows']['decisions']:,} targets; "
          f"{summary['table_rows']['evaluations']:,} single-order outcomes; {summary['runtime_seconds']:.2f}s; {target}", flush=True)


if __name__ == "__main__":
    main()
