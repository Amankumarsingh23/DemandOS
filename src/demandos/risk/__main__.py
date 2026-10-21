"""Run the bounded Season 21 SIMULATED / SCENARIO-ASSUMED experiment."""
import argparse
from pathlib import Path
from .pipeline import run,load_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path.cwd());p.add_argument("--config",type=Path)
    a=p.parse_args();summary,target=run(a.root,load_config(a.config or a.root/"configs/risk.toml"))
    print(f"SIMULATED risk experiment passed: {summary['pool_runs']} pool runs; {summary['daily_rows']:,} daily rows; {target}",flush=True)


if __name__=="__main__":main()
