"""Print a compact aggregate comparison of Stage B EHRSQL result JSON files.

Example:
    python harness/compare_ehrsql.py results/*stageb2*valid.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_result(path: Path) -> dict[str, Any]:
    """Load one Stage B result JSON file."""
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if result.get("stage") != "B" or result.get("dataset") != "EHRSQL v1.5.0 MIMIC-III":
        raise ValueError(f"Not a Stage B EHRSQL result: {path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage B comparison command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    return parser


def main() -> None:
    """Print fixed aggregate EHRSQL metrics for supplied result files."""
    args = build_parser().parse_args()
    results = [load_result(path) for path in args.results]
    print(
        "run name\tanswerable EX\tvalid SQL\tabstention accuracy\tunsafe answers"
        "\tunnecessary abstentions"
    )
    for result in results:
        print(
            f"{result['run_name']}\t"
            f"{result['answerable_ex']:.1%} ({result['n_correct']}/{result['n_scored']})\t"
            f"{result['valid_sql_rate']:.1%} ({result['n_valid']}/{result['n_answerable']})\t"
            f"{result['impossible_abstention_accuracy']:.1%} "
            f"({result['n_correct_abstentions']}/{result['n_impossible']})\t"
            f"{result['n_unsafe_answers']}\t{result['n_unnecessary_abstentions']}"
        )


if __name__ == "__main__":
    main()
