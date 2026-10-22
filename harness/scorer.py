"""Score predicted SQL with the frozen Stage A execution-accuracy rules.

Example:
    outcome = score_example(db_path, gold_sql, predicted_sql)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from harness.executor import run_sql


ERROR_TYPES = (
    "syntax_error",
    "no_such_table",
    "no_such_column",
    "timeout",
    "wrong_result",
    "empty_prediction",
    "other",
)


def _normalise_value(value: object) -> object:
    """Round SQLite float values to the frozen six-decimal precision."""
    return round(value, 6) if isinstance(value, float) else value


def _normalise_row(row: Sequence[object]) -> tuple[object, ...]:
    """Normalise every value in one SQLite result row."""
    return tuple(_normalise_value(value) for value in row)


def _row_sort_key(row: tuple[object, ...]) -> tuple[str, ...]:
    """Build a stable sort key that also handles mixed SQLite value types."""
    return tuple(f"{type(value).__name__}:{value!r}" for value in row)


def compare_results(
    gold_rows: Sequence[Sequence[object]], pred_rows: Sequence[Sequence[object]], ordered: bool
) -> bool:
    """Compare results in order or as an unordered multiset, as required by Stage A."""
    normalised_gold = [_normalise_row(row) for row in gold_rows]
    normalised_pred = [_normalise_row(row) for row in pred_rows]
    if ordered:
        return normalised_gold == normalised_pred
    return sorted(normalised_gold, key=_row_sort_key) == sorted(
        normalised_pred, key=_row_sort_key
    )


def has_order_by(sql: str) -> bool:
    """Return whether a gold SQL query contains an ORDER BY clause."""
    return bool(re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE))


def score_example(
    db_path: Path, gold_sql: str, pred_sql: str, timeout_s: float = 10
) -> dict[str, Any]:
    """Score one predicted query and retain failures needed for run reporting."""
    gold_result = run_sql(db_path, gold_sql, timeout_s)
    if not gold_result.ok:
        return {
            "correct": False,
            "valid": False,
            "error_type": gold_result.error_type,
            "gold_failed": True,
            "gold_error_message": gold_result.error_message,
        }

    pred_result = run_sql(db_path, pred_sql, timeout_s)
    if not pred_result.ok:
        return {
            "correct": False,
            "valid": False,
            "error_type": pred_result.error_type,
            "gold_failed": False,
            "error_message": pred_result.error_message,
        }

    correct = compare_results(gold_result.rows or [], pred_result.rows or [], has_order_by(gold_sql))
    return {
        "correct": correct,
        "valid": True,
        "error_type": None if correct else "wrong_result",
        "gold_failed": False,
    }


def _database_path(example: Mapping[str, Any], database_root: Path | None) -> Path:
    """Find an example's SQLite file from an explicit path or Spider database root."""
    if "db_path" in example:
        return Path(example["db_path"])
    if database_root is None:
        raise ValueError("Examples need db_path values or a database_root argument.")
    db_id = example["db_id"]
    return database_root / db_id / f"{db_id}.sqlite"


def score_dataset(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    timeout_s: float = 10,
    database_root: Path | None = None,
) -> dict[str, Any]:
    """Score aligned Spider examples and predictions, returning the Stage A summary."""
    if len(examples) != len(predictions):
        raise ValueError("Examples and predictions must have the same length.")

    error_counts = Counter({error_type: 0 for error_type in ERROR_TYPES})
    outcomes: list[dict[str, Any]] = []
    for example, prediction in zip(examples, predictions, strict=True):
        outcome = score_example(
            _database_path(example, database_root), example["query"], prediction, timeout_s
        )
        outcomes.append(outcome)
        if not outcome["gold_failed"] and outcome["error_type"] is not None:
            error_counts[outcome["error_type"]] += 1

    n_examples = len(outcomes)
    n_gold_failed = sum(outcome["gold_failed"] for outcome in outcomes)
    n_scored = n_examples - n_gold_failed
    n_correct = sum(outcome["correct"] for outcome in outcomes)
    n_valid = sum(outcome["valid"] for outcome in outcomes)
    return {
        "n_examples": n_examples,
        "n_gold_failed": n_gold_failed,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_valid": n_valid,
        "ex": n_correct / n_scored if n_scored else 0.0,
        "valid_sql_rate": n_valid / n_examples if n_examples else 0.0,
        "error_counts": dict(error_counts),
        "outcomes": outcomes,
    }
