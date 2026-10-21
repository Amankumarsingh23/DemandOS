"""Run the bounded SIMULATED causal and off-policy evaluation experiment."""
import argparse
from pathlib import Path
from .pipeline import run,load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd());parser.add_argument("--config",type=Path)
    args=parser.parse_args();summary,target=run(args.root,load_config(args.config or args.root/"configs/causal.toml"))
    print(f"MODEL-WORLD causal evaluation passed: {summary['logging_rows']} synthetic log rows; {target}",flush=True)


if __name__=="__main__":main()
