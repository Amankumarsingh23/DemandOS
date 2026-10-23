"""Bounded Season 24 demo, safe prerequisite inspection and independent audit."""
import argparse
import json
from pathlib import Path
import sys
from demandos.serving.core import Blocked, read_json
from .config import CONFIG, prerequisites


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-prereqs", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    ready = prerequisites(root)
    if args.check_prereqs or args.dry_run or not ready["ready"]:
        print(json.dumps(dict(prerequisites=ready, plan=CONFIG), indent=2))
        return 0 if ready["ready"] else 1
    try:
        if args.verify:
            from .audit import audit
            from demandos.serving.inputs import load
            base = root/"data/capstone/m5_h28"
            result = audit(base/read_json(base/"latest_verified.json")["path"], data=load(root))
        else:
            from .pipeline import run
            result = run(root)
        print(json.dumps(result, indent=2))
        return 0
    except (Blocked, FileNotFoundError, ValueError, AssertionError, KeyError) as exc:
        print(f"Capstone stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
