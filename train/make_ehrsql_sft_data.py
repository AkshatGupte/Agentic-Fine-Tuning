"""Build frozen-Prompt-B EHRSQL SFT records and report token lengths.

Run on Kaggle after attaching the private EHRSQL MIMIC-III dataset:
    python train/make_ehrsql_sft_data.py --config configs/train_ehrsql_qlora.yaml \
      --data-root /kaggle/input/datasets/akshatgupte/ehrsql-mimic-iii-stage-b
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import logging
import os
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ehrsql import database_path, load_ehrsql_split
from harness.ehrsql_prompts import build_ehrsql_prompt_b
from harness.schema import get_schema


LOGGER = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Minimal tokenizer interface used to calculate Stage B SFT lengths."""

    eos_token: str

    def __call__(self, text: str, *, add_special_tokens: bool) -> Any:
        """Tokenize text and return an object with ``input_ids``."""


def load_config(path: Path) -> dict[str, Any]:
    """Load the Stage B QLoRA configuration."""
    import yaml

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_record(example: dict[str, Any], schema: str) -> dict[str, str]:
    """Build one frozen-Prompt-B answerable EHRSQL completion record."""
    if bool(example["is_impossible"]):
        raise ValueError("Impossible EHRSQL examples must not enter SFT data.")
    return {
        "id": str(example["id"]),
        "prompt": build_ehrsql_prompt_b(schema, str(example["question"])),
        "completion": str(example["query"]),
    }


def token_length(record: dict[str, str], tokenizer: Tokenizer) -> int:
    """Return the exact non-truncated token length used by the QLoRA trainer."""
    text = record["prompt"] + record["completion"] + tokenizer.eos_token
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def build_parser() -> argparse.ArgumentParser:
    """Build the EHRSQL SFT preparation command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_ehrsql_qlora.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Prepare only the first 20 records.")
    return parser


def main() -> None:
    """Create answerable SFT JSONL and a reproducible token-length report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_root = args.data_root or Path(os.environ.get("DATA_ROOT", "data/raw/ehrsql/dataset/ehrsql"))
    output = args.output or Path(config["data_file"])
    max_length = args.max_length or int(config["max_seq_length"])
    today = date.today().isoformat()
    report = args.report or Path("results") / f"{today}_stageb3_ehrsql_mimic_iii_sft_prep.json"

    examples = load_ehrsql_split(data_root, "train")
    answerable = [example for example in examples if not example["is_impossible"]]
    if args.smoke:
        answerable = answerable[:20]
    if not answerable:
        raise ValueError("No answerable EHRSQL training examples were selected.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    schema = get_schema(database_path(data_root))
    kept: list[dict[str, str]] = []
    lengths: list[int] = []
    dropped_ids: list[str] = []
    for example in answerable:
        record = build_record(example, schema)
        length = token_length(record, tokenizer)
        lengths.append(length)
        if length <= max_length:
            kept.append(record)
        else:
            dropped_ids.append(record["id"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in kept:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {
        "run_name": "stageb3_ehrsql_mimic_iii_sft_prep" + ("_smoke" if args.smoke else ""),
        "date": today,
        "stage": "B",
        "step": "B3",
        "dataset": "EHRSQL v1.5.0 MIMIC-III",
        "split": "train",
        "prompt": "B",
        "model": config["model_name"],
        "max_seq_length": max_length,
        "n_source_examples": len(examples),
        "n_answerable_examples": len(answerable),
        "n_kept": len(kept),
        "n_dropped": len(dropped_ids),
        "token_length": {
            "min": min(lengths),
            "median": median(lengths),
            "mean": mean(lengths),
            "max": max(lengths),
        },
        "dropped_ids": dropped_ids,
        "output": str(output),
        "notes": "No model training was performed by this preparation run.",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"kept={result['n_kept']} dropped={result['n_dropped']} "
        f"token_min={result['token_length']['min']} "
        f"token_median={result['token_length']['median']} "
        f"token_mean={result['token_length']['mean']:.1f} "
        f"token_max={result['token_length']['max']}"
    )
    LOGGER.info("Saved SFT JSONL to %s", output)
    LOGGER.info("Saved preparation report to %s", report)


if __name__ == "__main__":
    main()
