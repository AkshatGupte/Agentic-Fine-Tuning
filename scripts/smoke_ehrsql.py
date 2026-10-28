"""Smoke-test Stage B EHRSQL MIMIC-III loading and scoring without a model.

Run locally:
    python scripts/smoke_ehrsql.py --config configs/ehrsql.yaml
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

from data.ehrsql import database_path, load_ehrsql_split
from harness.ehrsql_scorer import ABSTAIN_TOKEN, score_ehrsql_dataset


LOGGER = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    """Load the Stage B YAML configuration."""
    import yaml

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _git_commit() -> str | None:
    """Return the current short Git revision when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _select_examples(examples: list[dict[str, Any]], answerable_limit: int) -> list[dict[str, Any]]:
    """Select a tiny deterministic mix of answerable and impossible records."""
    answerable = [example for example in examples if not example["is_impossible"]][:answerable_limit]
    impossible = next((example for example in examples if example["is_impossible"]), None)
    if impossible is not None:
        answerable.append(impossible)
    if not answerable:
        raise ValueError("No EHRSQL examples selected for smoke testing.")
    return answerable


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Stage B smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ehrsql.yaml"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="EHRSQL root; default is DATA_ROOT or the YAML data_root.",
    )
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--answerable-limit", type=int, default=3)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser


def main() -> None:
    """Validate a small EHRSQL set by scoring oracle SQL and abstentions."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    if args.answerable_limit < 1:
        raise ValueError("--answerable-limit must be at least 1.")
    config = _load_config(args.config)
    data_root = args.data_root or Path(os.environ.get("DATA_ROOT", config["paths"]["data_root"]))
    results_dir = args.results_dir or Path(config["paths"]["results_dir"])
    timeout_s = float(config["sql"]["timeout_seconds"])
    examples = _select_examples(load_ehrsql_split(data_root, args.split), args.answerable_limit)
    predictions = [ABSTAIN_TOKEN if example["is_impossible"] else example["query"] for example in examples]
    summary = score_ehrsql_dataset(examples, predictions, database_path(data_root), timeout_s)

    run_name = "stageb1_ehrsql_mimic_iii_smoke"
    today = date.today().isoformat()
    result = {
        "run_name": run_name,
        "date": today,
        "stage": "B",
        "step": "B1",
        "dataset": "EHRSQL v1.5.0 MIMIC-III",
        "model": None,
        "adapter": None,
        "prompt": None,
        "split": args.split,
        **{key: value for key, value in summary.items() if key != "outcomes"},
        "settings": {
            "seed": config["seed"],
            "timeout_s": timeout_s,
            "answerable_limit": args.answerable_limit,
            "oracle_predictions": True,
        },
        "git_commit": _git_commit(),
        "notes": "Smoke test used gold SQL for answerable examples and ABSTAIN for impossible examples.",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{today}_{run_name}.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")

    predictions_path = results_dir / "predictions" / f"{run_name}.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as file:
        for example, prediction, outcome in zip(examples, predictions, summary["outcomes"], strict=True):
            file.write(
                json.dumps(
                    {
                        "id": example["id"],
                        "question": example["question"],
                        "gold_sql": example.get("query"),
                        "predicted_sql": prediction,
                        "is_impossible": example["is_impossible"],
                        "correct": outcome["correct"],
                        "valid": outcome["valid"],
                        "error_type": outcome["error_type"],
                        "gold_failed": outcome["gold_failed"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"{run_name}: answerable EX {summary['answerable_ex']:.1%} "
        f"({summary['n_correct']}/{summary['n_scored']}), valid SQL "
        f"{summary['valid_sql_rate']:.1%}, abstention accuracy "
        f"{summary['impossible_abstention_accuracy']:.1%} "
        f"({summary['n_correct_abstentions']}/{summary['n_impossible']})"
    )
    LOGGER.info("Saved result to %s", result_path)


if __name__ == "__main__":
    main()
