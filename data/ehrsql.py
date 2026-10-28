"""Load and validate the frozen Stage B EHRSQL MIMIC-III split files.

Example:
    examples = load_ehrsql_split(Path("data/raw/ehrsql/dataset/ehrsql"), "valid")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


EHRSQL_SPLIT = Literal["train", "valid", "test"]
MIMIC_III_DB_ID = "mimic_iii"


def database_path(data_root: Path) -> Path:
    """Return the MIMIC-III database from official nested or flat Kaggle layout."""
    candidates = (
        data_root / MIMIC_III_DB_ID / f"{MIMIC_III_DB_ID}.sqlite",
        data_root / f"{MIMIC_III_DB_ID}.sqlite",
    )
    for path in candidates:
        if path.is_file():
            return path
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"EHRSQL MIMIC-III SQLite database not found. Tried: {attempted}")


def _split_path(data_root: Path, split: EHRSQL_SPLIT) -> Path:
    """Find one split file from official nested or flat Kaggle layout."""
    candidates = (
        data_root / MIMIC_III_DB_ID / f"{split}.json",
        data_root / f"{split}.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"EHRSQL split file not found. Tried: {attempted}")


def _validate_example(example: Any, split: EHRSQL_SPLIT, index: int) -> dict[str, Any]:
    """Validate one official EHRSQL MIMIC-III record."""
    if not isinstance(example, dict):
        raise ValueError(f"{split}.json record {index} must be an object.")
    required = ("id", "db_id", "question", "split", "is_impossible")
    missing = [field for field in required if field not in example]
    if missing:
        raise ValueError(f"{split}.json record {index} is missing fields: {missing}")
    if example["db_id"] != MIMIC_III_DB_ID:
        raise ValueError(f"{split}.json record {index} has unexpected db_id: {example['db_id']!r}")
    if example["split"] != split:
        raise ValueError(
            f"{split}.json record {index} has split {example['split']!r}, expected {split!r}."
        )
    if not isinstance(example["question"], str) or not example["question"].strip():
        raise ValueError(f"{split}.json record {index} has no usable question.")
    if not isinstance(example["is_impossible"], bool):
        raise ValueError(f"{split}.json record {index} has a non-boolean is_impossible value.")
    if not example["is_impossible"]:
        query = example.get("query")
        if not isinstance(query, str) or not query.strip() or query.lower() == "null":
            raise ValueError(f"{split}.json record {index} has no usable answerable SQL query.")
    return example


def load_ehrsql_split(data_root: Path, split: EHRSQL_SPLIT) -> list[dict[str, Any]]:
    """Load one official EHRSQL MIMIC-III split and validate its records."""
    database_path(data_root)
    split_path = _split_path(data_root, split)
    with split_path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"EHRSQL split must be a non-empty JSON list: {split_path}")
    return [_validate_example(record, split, index) for index, record in enumerate(records)]
