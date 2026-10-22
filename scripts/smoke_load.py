"""Load a prepared Spider split and print one example with its SQLite schema.

Run after preparing splits:
    .venv/bin/python scripts/smoke_load.py --data-root data/raw/spider
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.schema import get_schema


def load_split(split_path: Path) -> list[dict[str, Any]]:
    """Load a prepared Stage A split."""
    with split_path.open(encoding="utf-8") as file:
        return json.load(file)


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke-test command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data/raw/spider")),
        help="Spider root (default: DATA_ROOT or data/raw/spider).",
    )
    parser.add_argument(
        "--split-path", type=Path, default=Path("data/splits/dev_eval.json")
    )
    return parser


def main() -> None:
    """Print one dev evaluation example and its schema."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    examples = load_split(args.split_path)
    if not examples:
        raise ValueError(f"No examples found in {args.split_path}")

    example = examples[0]
    db_path = args.data_root / "database" / example["db_id"] / f"{example['db_id']}.sqlite"
    schema = get_schema(db_path)
    print(json.dumps(example, ensure_ascii=False, indent=2))
    print("\n### Schema\n")
    print(schema)


if __name__ == "__main__":
    main()
