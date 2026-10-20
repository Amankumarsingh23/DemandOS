"""Run SCENARIO-ASSUMED constrained allocation and coordinated inventory simulation."""
import argparse
from pathlib import Path
from .pipeline import run,load_config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path.cwd());p.add_argument("--config",type=Path)
    args=p.parse_args();s,target=run(args.root,load_config(args.config or args.root/"configs/optimization.toml"))
    print(f"SIMULATED optimization passed: {s['pool_runs']} pool runs; {s['daily_rows']:,} daily rows; {s['runtime_seconds']:.2f}s; {target}",flush=True)


if __name__=="__main__":main()
