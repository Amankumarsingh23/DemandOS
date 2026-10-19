"""Train and verify a compact global point TFT on the real M5 cohort."""

import argparse
from pathlib import Path
from .config import load_config
from .pipeline import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--config",type=Path)
    args=parser.parse_args()
    summary,output=run(args.root,load_config(args.config or args.root/"configs/tft.toml"))
    print(f"TFT verification passed: {summary['table_rows']['forecasts']:,} forecast rows; {summary['runtime_seconds']:.2f}s; {output}",flush=True)


if __name__=="__main__": main()
