"""Run one Stage B EHRSQL prompt-only baseline and save reproducible artifacts.

Example Kaggle command:
    python harness/run_ehrsql_baseline.py --config configs/ehrsql.yaml --prompt A \
      --data-root /kaggle/input/ehrsql-mimic-iii-stage-b --load-in-4bit
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ehrsql import database_path, load_ehrsql_split
from harness.ehrsql_output import clean_ehrsql_output
from harness.ehrsql_prompts import build_ehrsql_prompt_a, build_ehrsql_prompt_b
from harness.ehrsql_scorer import score_ehrsql_dataset
from harness.generate import generate_sql, load_model_and_tokenizer
from harness.schema import get_schema


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set available Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def load_config(path: Path) -> dict[str, Any]:
    """Load the Stage B YAML configuration."""
    import yaml

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def git_commit() -> str | None:
    """Return the short Git revision if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _select_smoke_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose five answerable records plus one impossible record when available."""
    selected = [example for example in examples if not example["is_impossible"]][:5]
    impossible = next((example for example in examples if example["is_impossible"]), None)
    if impossible is not None:
        selected.append(impossible)
    return selected


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage B EHRSQL baseline command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ehrsql.yaml"))
    parser.add_argument("--prompt", choices=("A", "B"), required=True)
    parser.add_argument("--model", default=None, help="Override model_name in the YAML config.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Generate six tiny CPU-test examples.")
    return parser


def main() -> None:
    """Generate, score, and save one EHRSQL baseline run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    config = load_config(args.config)
    model_name = args.model or config["model_name"]
    if not model_name:
        raise ValueError("Set model_name in the config or pass --model.")
    data_root = args.data_root or Path(os.environ.get("DATA_ROOT", config["paths"]["data_root"]))
    results_dir = args.results_dir or Path(config["paths"]["results_dir"])
    timeout_s = float(config["sql"]["timeout_seconds"])
    generation = config["generation"]
    gold_failure_ids = set(config["evaluation"]["gold_failure_ids"])
    set_seed(int(config["seed"]))

    examples = load_ehrsql_split(data_root, args.split)
    if args.smoke:
        examples = _select_smoke_examples(examples)
    elif args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("No EHRSQL examples selected for baseline scoring.")

    prompt_builders: dict[str, Callable[[str, str], str]] = {
        "A": build_ehrsql_prompt_a,
        "B": build_ehrsql_prompt_b,
    }
    schema = get_schema(database_path(data_root))
    prompts = [prompt_builders[args.prompt](schema, example["question"]) for example in examples]
    model, tokenizer = load_model_and_tokenizer(model_name, args.load_in_4bit)
    raw_predictions = generate_sql(
        prompts,
        model,
        tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=int(generation["max_new_tokens"]),
    )
    predictions = [clean_ehrsql_output(text) for text in raw_predictions]
    summary = score_ehrsql_dataset(
        examples,
        predictions,
        database_path(data_root),
        timeout_s,
        known_gold_failure_ids=gold_failure_ids,
    )

    short_model = model_name.rsplit("/", maxsplit=1)[-1].lower().replace("-instruct", "")
    run_name = f"stageb2_ehrsql_mimic_iii_prompt{args.prompt}_{short_model}_{args.split}"
    if args.smoke:
        run_name += "_smoke"
    today = date.today().isoformat()
    result = {
        "run_name": run_name,
        "date": today,
        "stage": "B",
        "step": "B2",
        "dataset": "EHRSQL v1.5.0 MIMIC-III",
        "model": model_name,
        "adapter": None,
        "prompt": args.prompt,
        "split": args.split,
        "gold_failure_ids": sorted(gold_failure_ids),
        **{key: value for key, value in summary.items() if key != "outcomes"},
        "settings": {
            "seed": config["seed"],
            "max_new_tokens": generation["max_new_tokens"],
            "greedy": generation["greedy"],
            "batch_size": args.batch_size,
            "load_in_4bit": args.load_in_4bit,
            "timeout_s": timeout_s,
            "fixed_gold_failure_count": len(gold_failure_ids),
        },
        "git_commit": git_commit(),
        "notes": (
            "Smoke run; fixed gold failures excluded."
            if args.smoke
            else "Fixed EHRSQL gold-query preflight failures excluded from EX."
        ),
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
                        "unsafe_answer": outcome["unsafe_answer"],
                        "unnecessary_abstention": outcome["unnecessary_abstention"],
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
        f"({summary['n_correct_abstentions']}/{summary['n_impossible']}), "
        f"unsafe answers {summary['n_unsafe_answers']}, unnecessary abstentions "
        f"{summary['n_unnecessary_abstentions']}"
    )
    LOGGER.info("Saved result to %s", result_path)


if __name__ == "__main__":
    main()
