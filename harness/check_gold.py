"""Validate Spider gold SQL with the Stage A execution-accuracy scorer.

Run all dev examples:
    python harness/check_gold.py --data-root data/raw/spider

Run a CPU smoke test:
    python harness/check_gold.py --data-root data/raw/spider --smoke
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.scorer import score_dataset


LOGGER = logging.getLogger(__name__)


def _git_commit() -> str | None:
    """Return the current short commit hash when this is a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_examples(split_path: Path) -> list[dict[str, Any]]:
    """Load a prepared Spider split from disk."""
    with split_path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_predictions(
    output_path: Path, examples: list[dict[str, Any]], outcomes: list[dict[str, Any]]
) -> None:
    """Save per-example gold-as-prediction outcomes as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for example, outcome in zip(examples, outcomes, strict=True):
            record = {
                "question": example["question"],
                "gold_sql": example["query"],
                "predicted_sql": example["query"],
                "correct": outcome["correct"],
                "error_type": outcome["error_type"],
                "gold_failed": outcome["gold_failed"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for gold-query validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data/raw/spider")),
        help="Spider root (default: DATA_ROOT or data/raw/spider).",
    )
    parser.add_argument("--split-path", type=Path, default=Path("data/splits/dev_eval.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--timeout-s", type=float, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Score only three examples.")
    return parser


def main() -> None:
    """Score gold SQL as predictions and save a reproducible Step 2 result."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    examples = _load_examples(args.split_path)
    limit = 3 if args.smoke else args.limit
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError("No examples selected for gold validation.")

    summary = score_dataset(
        examples,
        [example["query"] for example in examples],
        timeout_s=args.timeout_s,
        database_root=args.data_root / "database",
    )
    run_name = "step2_check_gold_smoke" if args.smoke else "step2_check_gold"
    today = date.today().isoformat()
    result = {
        "run_name": run_name,
        "date": today,
        "step": 2,
        "model": None,
        "adapter": None,
        "prompt": None,
        "split": "dev_eval",
        "n_examples": summary["n_examples"],
        "n_gold_failed": summary["n_gold_failed"],
        "n_correct": summary["n_correct"],
        "n_scored": summary["n_scored"],
        "ex": summary["ex"],
        "valid_sql_rate": summary["valid_sql_rate"],
        "error_counts": summary["error_counts"],
        "settings": {"seed": 42, "timeout_s": args.timeout_s, "gold_as_prediction": True},
        "git_commit": _git_commit(),
        "notes": "Gold SQL was used as each prediction.",
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"{today}_step2_{run_name.removeprefix('step2_')}.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    _write_predictions(args.results_dir / "predictions" / f"{run_name}.jsonl", examples, summary["outcomes"])

    print(
        f"Gold check: EX {summary['ex']:.1%} "
        f"({summary['n_correct']}/{summary['n_scored']}), "
        f"valid SQL {summary['valid_sql_rate']:.1%}, "
        f"gold failed {summary['n_gold_failed']}"
    )
    LOGGER.info("Saved result to %s", result_path)


if __name__ == "__main__":
    main()
