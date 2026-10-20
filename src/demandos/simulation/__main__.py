"""Run sequential SCENARIO-BASED M5 inventory backtests and bounded sensitivity."""
import argparse
from pathlib import Path
from .pipeline import load_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary, target = run(args.root, load_config(args.config or args.root/"configs/simulation.toml"))
    print(f"SIMULATED sequential verification passed: {summary['main_runs']:,} main runs; "
          f"{summary['main_daily_rows']:,} main daily rows; {summary['runtime_seconds']:.2f}s; {target}", flush=True)


if __name__ == "__main__": main()
