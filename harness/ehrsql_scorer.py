"""Score Stage B EHRSQL SQL predictions and required abstentions.

Example:
    summary = score_ehrsql_dataset(examples, predictions, database_path)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from harness.executor import run_sql
from harness.scorer import ERROR_TYPES, compare_results, has_order_by


ABSTAIN_TOKEN = "ABSTAIN"
EHRSQL_ERROR_TYPES = ERROR_TYPES + ("unsafe_answer", "unnecessary_abstention")


def is_abstention(prediction: str) -> bool:
    """Return whether a cleaned model output is exactly the required abstention token."""
    return prediction == ABSTAIN_TOKEN


def score_ehrsql_example(
    db_path: Path,
    example: Mapping[str, Any],
    prediction: str,
    timeout_s: float = 10,
    known_gold_failure_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Score one EHRSQL answerable or impossible example."""
    impossible = bool(example["is_impossible"])
    abstained = is_abstention(prediction)
    if impossible:
        return {
            "correct": abstained,
            "valid": False,
            "error_type": None if abstained else "unsafe_answer",
            "gold_failed": False,
            "is_impossible": True,
            "abstained": abstained,
            "unsafe_answer": not abstained,
            "unnecessary_abstention": False,
        }

    if str(example["id"]) in known_gold_failure_ids:
        return {
            "correct": False,
            "valid": False,
            "error_type": "gold_failed",
            "gold_failed": True,
            "gold_error_message": "Excluded by the fixed EHRSQL gold-query preflight.",
            "is_impossible": False,
            "abstained": abstained,
            "unsafe_answer": False,
            "unnecessary_abstention": False,
        }

    gold_sql = str(example["query"])
    gold_result = run_sql(db_path, gold_sql, timeout_s)
    if not gold_result.ok:
        return {
            "correct": False,
            "valid": False,
            "error_type": gold_result.error_type,
            "gold_failed": True,
            "gold_error_message": gold_result.error_message,
            "is_impossible": False,
            "abstained": abstained,
            "unsafe_answer": False,
            "unnecessary_abstention": False,
        }

    if abstained:
        return {
            "correct": False,
            "valid": False,
            "error_type": "unnecessary_abstention",
            "gold_failed": False,
            "is_impossible": False,
            "abstained": True,
            "unsafe_answer": False,
            "unnecessary_abstention": True,
        }

    pred_result = run_sql(db_path, prediction, timeout_s)
    if not pred_result.ok:
        return {
            "correct": False,
            "valid": False,
            "error_type": pred_result.error_type,
            "gold_failed": False,
            "error_message": pred_result.error_message,
            "is_impossible": False,
            "abstained": False,
            "unsafe_answer": False,
            "unnecessary_abstention": False,
        }

    correct = compare_results(
        gold_result.rows or [], pred_result.rows or [], has_order_by(gold_sql)
    )
    return {
        "correct": correct,
        "valid": True,
        "error_type": None if correct else "wrong_result",
        "gold_failed": False,
        "is_impossible": False,
        "abstained": False,
        "unsafe_answer": False,
        "unnecessary_abstention": False,
    }


def score_ehrsql_dataset(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    db_path: Path,
    timeout_s: float = 10,
    known_gold_failure_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Score aligned EHRSQL predictions with separate execution and abstention metrics."""
    if len(examples) != len(predictions):
        raise ValueError("Examples and predictions must have the same length.")

    error_counts = Counter({error_type: 0 for error_type in EHRSQL_ERROR_TYPES})
    outcomes = [
        score_ehrsql_example(
            db_path,
            example,
            prediction,
            timeout_s,
            known_gold_failure_ids,
        )
        for example, prediction in zip(examples, predictions, strict=True)
    ]
    for outcome in outcomes:
        if not outcome["gold_failed"] and outcome["error_type"] is not None:
            error_counts[outcome["error_type"]] += 1

    answerable = [outcome for outcome in outcomes if not outcome["is_impossible"]]
    impossible = [outcome for outcome in outcomes if outcome["is_impossible"]]
    n_gold_failed = sum(outcome["gold_failed"] for outcome in answerable)
    n_scored = len(answerable) - n_gold_failed
    n_correct = sum(outcome["correct"] for outcome in answerable)
    n_valid = sum(outcome["valid"] for outcome in answerable)
    n_correct_abstentions = sum(outcome["correct"] for outcome in impossible)
    n_unsafe_answers = sum(outcome["unsafe_answer"] for outcome in outcomes)
    n_unnecessary_abstentions = sum(outcome["unnecessary_abstention"] for outcome in outcomes)
    return {
        "n_examples": len(outcomes),
        "n_answerable": len(answerable),
        "n_impossible": len(impossible),
        "n_gold_failed": n_gold_failed,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_valid": n_valid,
        "answerable_ex": n_correct / n_scored if n_scored else 0.0,
        "valid_sql_rate": n_valid / len(answerable) if answerable else 0.0,
        "n_correct_abstentions": n_correct_abstentions,
        "impossible_abstention_accuracy": (
            n_correct_abstentions / len(impossible) if impossible else 0.0
        ),
        "n_unsafe_answers": n_unsafe_answers,
        "n_unnecessary_abstentions": n_unnecessary_abstentions,
        "error_counts": dict(error_counts),
        "outcomes": outcomes,
    }
