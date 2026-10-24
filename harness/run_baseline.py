"""Run and score a prompt-only Stage A SQL baseline.

Example Kaggle command:
    python harness/run_baseline.py --config configs/base.yaml --prompt A \
      --data-root /kaggle/input/datasets/akshatgupte/spider-data-2
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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.generate import generate_sql, load_model_and_tokenizer
from harness.prompts import build_prompt_a, build_prompt_b, clean_sql_output
from harness.schema import get_schema
from harness.scorer import score_dataset


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
    """Load the shared YAML configuration only when a baseline is run."""
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


def build_parser() -> argparse.ArgumentParser:
    """Build the baseline-run command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--prompt", choices=("A", "B"), required=True)
    parser.add_argument("--model", default=None, help="Override model_name in the YAML config.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=Path("data/splits/dev_eval.json"))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Generate and score only five examples.")
    parser.add_argument("--step", type=int, default=3, help="Stage step recorded in output names.")
    return parser


def main() -> None:
    """Generate, clean, score, and save one prompt-only baseline run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    model_name = args.model or config["model_name"]
    if not model_name:
        raise ValueError("Set model_name in the config or pass --model.")
    data_root = args.data_root or Path(os.environ.get("DATA_ROOT", paths["data_root"]))
    results_dir = args.results_dir or Path(paths["results_dir"])
    timeout_s = config["sql"]["timeout_seconds"]
    seed = config["seed"]
    set_seed(seed)

    with args.split_path.open(encoding="utf-8") as file:
        examples: list[dict[str, Any]] = json.load(file)
    limit = 5 if args.smoke else args.limit
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError("No examples selected for baseline scoring.")

    prompt_builder = build_prompt_a if args.prompt == "A" else build_prompt_b
    prompts = []
    for example in examples:
        db_id = example["db_id"]
        schema = get_schema(data_root / "database" / db_id / f"{db_id}.sqlite")
        prompts.append(prompt_builder(schema, example["question"]))
    model, tokenizer = load_model_and_tokenizer(model_name, args.load_in_4bit)
    predictions = [clean_sql_output(text) for text in generate_sql(prompts, model, tokenizer, args.batch_size)]
    summary = score_dataset(
        examples,
        predictions,
        timeout_s=timeout_s,
        database_root=data_root / "database",
    )

    short_model = model_name.rsplit("/", maxsplit=1)[-1].lower().replace("-instruct", "")
    run_name = f"step{args.step}_prompt{args.prompt}_{short_model}"
    if args.smoke:
        run_name += "_smoke"
    today = date.today().isoformat()
    result = {
        "run_name": run_name,
        "date": today,
        "step": args.step,
        "model": model_name,
        "adapter": None,
        "prompt": args.prompt,
        "split": "dev_eval",
        "n_examples": summary["n_examples"],
        "n_gold_failed": summary["n_gold_failed"],
        "n_correct": summary["n_correct"],
        "n_scored": summary["n_scored"],
        "ex": summary["ex"],
        "valid_sql_rate": summary["valid_sql_rate"],
        "error_counts": summary["error_counts"],
        "settings": {
            "seed": seed,
            "max_new_tokens": 256,
            "greedy": True,
            "batch_size": args.batch_size,
            "load_in_4bit": args.load_in_4bit,
        },
        "git_commit": git_commit(),
        "notes": "Smoke run" if args.smoke else "",
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
                        "question": example["question"],
                        "gold_sql": example["query"],
                        "predicted_sql": prediction,
                        "correct": outcome["correct"],
                        "error_type": outcome["error_type"],
                        "gold_failed": outcome["gold_failed"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"{run_name}: EX {summary['ex']:.1%} ({summary['n_correct']}/{summary['n_scored']}), "
        f"valid SQL {summary['valid_sql_rate']:.1%}"
    )
    LOGGER.info("Saved result to %s", result_path)


if __name__ == "__main__":
    main()
