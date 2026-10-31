"""Validate Stage B EHRSQL gold SQL and save the fixed exclusion artifact.

Run locally or on Kaggle:
    python harness/check_ehrsql_gold.py --config configs/ehrsql.yaml \
      --data-root data/raw/ehrsql/dataset/ehrsql
"""

from __future__ import annotations

import argparse
from collections import Counter
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


def build_parser() -> argparse.ArgumentParser:
    """Build the gold-preflight command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ehrsql.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser


def main() -> None:
    """Run gold SQL and oracle abstentions to establish fixed gold failures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    config = _load_config(args.config)
    data_root = args.data_root or Path(os.environ.get("DATA_ROOT", config["paths"]["data_root"]))
    results_dir = args.results_dir or Path(config["paths"]["results_dir"])
    timeout_s = float(config["sql"]["timeout_seconds"])
    examples = load_ehrsql_split(data_root, args.split)
    predictions = [ABSTAIN_TOKEN if example["is_impossible"] else example["query"] for example in examples]
    summary = score_ehrsql_dataset(examples, predictions, database_path(data_root), timeout_s)
    gold_failure_ids = [
        example["id"]
        for example, outcome in zip(examples, summary["outcomes"], strict=True)
        if outcome["gold_failed"]
    ]
    gold_failure_error_counts = Counter(
        outcome["error_type"]
        for outcome in summary["outcomes"]
        if outcome["gold_failed"] and outcome["error_type"] is not None
    )

    run_name = f"stageb2_ehrsql_mimic_iii_check_gold_{args.split}"
    today = date.today().isoformat()
    result = {
        "run_name": run_name,
        "date": today,
        "stage": "B",
        "step": "B2",
        "dataset": "EHRSQL v1.5.0 MIMIC-III",
        "model": None,
        "adapter": None,
        "prompt": None,
        "split": args.split,
        **{key: value for key, value in summary.items() if key != "outcomes"},
        "gold_failure_ids": gold_failure_ids,
        "gold_failure_error_counts": dict(gold_failure_error_counts),
        "settings": {"seed": config["seed"], "timeout_s": timeout_s, "gold_as_prediction": True},
        "git_commit": _git_commit(),
        "notes": "Gold SQL was used for answerable examples; ABSTAIN was used for impossible examples.",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{today}_{run_name}.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    prediction_path = results_dir / "predictions" / f"{run_name}.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as file:
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
        f"{run_name}: gold failures {summary['n_gold_failed']}, "
        f"answerable EX {summary['answerable_ex']:.1%} "
        f"({summary['n_correct']}/{summary['n_scored']})"
    )
    LOGGER.info("Saved result to %s", result_path)


if __name__ == "__main__":
    main()
