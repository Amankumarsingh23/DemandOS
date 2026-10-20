"""Run nested historical models and bounded real M5 conformal evaluation."""
import argparse
from pathlib import Path
from .pipeline import run, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary, output = run(args.root, load_config(args.config or args.root/"configs/conformal.toml"))
    print(f"Conformal verification passed: {summary['table_rows']['intervals']:,} intervals; {summary['runtime_seconds']:.2f}s; {output}", flush=True)


if __name__ == "__main__":
    main()
