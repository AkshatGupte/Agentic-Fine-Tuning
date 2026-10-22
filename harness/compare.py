"""Print a compact comparison table from saved Stage A result JSON files.

Run:
    python harness/compare.py results/*_step3_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the result-comparison command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_files", nargs="+", type=Path)
    return parser


def main() -> None:
    """Load result files and print run name, EX count, and valid-SQL rate."""
    args = build_parser().parse_args()
    results = []
    for path in args.result_files:
        with path.open(encoding="utf-8") as file:
            results.append(json.load(file))
    print("| Run | EX | Valid SQL |")
    print("|---|---:|---:|")
    for result in results:
        total = result.get("n_scored", result["n_examples"] - result["n_gold_failed"])
        correct = result.get("n_correct", round(result["ex"] * total))
        print(
            f"| {result['run_name']} | {result['ex']:.1%} ({correct}/{total}) "
            f"| {result['valid_sql_rate']:.1%} |"
        )


if __name__ == "__main__":
    main()
