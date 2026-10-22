"""Run the production-oriented LOCAL historical serving demonstration."""
import argparse
from pathlib import Path
from .pipeline import run,load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd());parser.add_argument("--config",type=Path)
    args=parser.parse_args();summary,path=run(args.root,load_config(args.config or args.root/"configs/serving.toml"))
    print(f"LOCAL historical serving PASS: {summary['forecast_rows']} forecasts; {path}",flush=True)


if __name__=="__main__":main()
