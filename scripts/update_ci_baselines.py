from __future__ import annotations

import argparse
from pathlib import Path

from benchmark import update_ci_baselines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    update_ci_baselines(args.source, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
