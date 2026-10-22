"""Create reproducible Stage A Spider train and evaluation splits.

Run locally:
    .venv/bin/python data/prepare_spider.py --data-root data/raw/spider

Run on Kaggle:
    DATA_ROOT=/kaggle/input/datasets/akshatgupte/spider-data-2 \\
      python data/prepare_spider.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZE = 3000
DEFAULT_DEV_SIZE = 300


def load_examples(data_root: Path, filename: str) -> list[dict[str, Any]]:
    """Load one Spider JSON split from ``data_root``."""
    path = data_root / filename
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def sample_examples(
    examples: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    """Return a deterministic random sample without modifying ``examples``."""
    if sample_size > len(examples):
        raise ValueError(
            f"Cannot sample {sample_size} examples from a split of {len(examples)}"
        )
    return random.Random(seed).sample(examples, sample_size)


def save_examples(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Save examples as readable UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(examples, file, ensure_ascii=False, indent=2)
        file.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for split preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data/raw/spider")),
        help="Spider root (default: DATA_ROOT or data/raw/spider).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for the saved deterministic splits.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--dev-size", type=int, default=DEFAULT_DEV_SIZE)
    return parser


def main() -> None:
    """Load Spider, save its fixed Stage A samples, and show one example."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    train_examples = load_examples(args.data_root, "train_spider.json")
    dev_examples = load_examples(args.data_root, "dev.json")

    train_sft = sample_examples(train_examples, args.train_size, args.seed)
    dev_eval = sample_examples(dev_examples, args.dev_size, args.seed)
    save_examples(train_sft, args.output_dir / "train_sft.json")
    save_examples(dev_eval, args.output_dir / "dev_eval.json")

    LOGGER.info("Saved train_sft: %d examples", len(train_sft))
    LOGGER.info("Saved dev_eval: %d examples", len(dev_eval))
    LOGGER.info("Example: %s", json.dumps(dev_eval[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
