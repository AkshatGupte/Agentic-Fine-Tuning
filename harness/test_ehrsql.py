"""Unit tests for Stage B EHRSQL loading, abstention, and execution scoring.

Run with:
    pytest harness/test_ehrsql.py
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from data.ehrsql import load_ehrsql_split
from harness.ehrsql_output import clean_ehrsql_output
from harness.ehrsql_scorer import ABSTAIN_TOKEN, score_ehrsql_dataset


def _write_split(root: Path, split: str, records: list[dict[str, object]]) -> None:
    """Write a minimal official-style EHRSQL split."""
    directory = root / "mimic_iii"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{split}.json").write_text(json.dumps(records), encoding="utf-8")


def _record(split: str, *, impossible: bool = False) -> dict[str, object]:
    """Build one valid minimal EHRSQL record."""
    return {
        "id": f"{split}-{impossible}",
        "db_id": "mimic_iii",
        "question": "Is this supported?",
        "query": "null" if impossible else "SELECT id FROM records",
        "split": split,
        "is_impossible": impossible,
    }


def test_loader_validates_split_labels_and_database() -> None:
    """The loader rejects records whose embedded split disagrees with their file."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        db_path = root / "mimic_iii" / "mimic_iii.sqlite"
        db_path.parent.mkdir(parents=True)
        sqlite3.connect(db_path).close()
        _write_split(root, "valid", [_record("train")])
        with pytest.raises(ValueError, match="expected 'valid'"):
            load_ehrsql_split(root, "valid")


def test_loader_supports_flat_kaggle_upload_layout() -> None:
    """The loader accepts a Kaggle dataset that stores database and splits at root."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with sqlite3.connect(root / "mimic_iii.sqlite"):
            pass
        (root / "valid.json").write_text(json.dumps([_record("valid")]), encoding="utf-8")
        examples = load_ehrsql_split(root, "valid")
        assert len(examples) == 1
        assert examples[0]["db_id"] == "mimic_iii"


def test_ehrsql_scores_execution_and_correct_abstention() -> None:
    """Gold SQL and exact ABSTAIN score correctly in their separate metric groups."""
    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / "test.sqlite"
        with sqlite3.connect(db_path) as connection:
            connection.execute("CREATE TABLE records (id INTEGER)")
            connection.execute("INSERT INTO records VALUES (1)")
        examples = [_record("valid"), _record("valid", impossible=True)]
        summary = score_ehrsql_dataset(
            examples, ["SELECT id FROM records", ABSTAIN_TOKEN], db_path
        )
        assert summary["answerable_ex"] == 1.0
        assert summary["valid_sql_rate"] == 1.0
        assert summary["impossible_abstention_accuracy"] == 1.0
        assert summary["n_unsafe_answers"] == 0


def test_ehrsql_scores_unsafe_and_unnecessary_outputs() -> None:
    """SQL for impossible questions and ABSTAIN for answerable questions are tracked."""
    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / "test.sqlite"
        with sqlite3.connect(db_path) as connection:
            connection.execute("CREATE TABLE records (id INTEGER)")
        examples = [_record("valid"), _record("valid", impossible=True)]
        summary = score_ehrsql_dataset(
            examples, [ABSTAIN_TOKEN, "SELECT id FROM records"], db_path
        )
        assert summary["n_unnecessary_abstentions"] == 1
        assert summary["n_unsafe_answers"] == 1
        assert summary["error_counts"]["unnecessary_abstention"] == 1
        assert summary["error_counts"]["unsafe_answer"] == 1


def test_gold_failure_takes_priority_over_answerable_abstention() -> None:
    """A broken gold query is excluded even when the prediction is ABSTAIN."""
    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / "test.sqlite"
        with sqlite3.connect(db_path) as connection:
            connection.execute("CREATE TABLE records (id INTEGER)")
        example = _record("valid")
        example["query"] = "SELECT missing_column FROM records"
        summary = score_ehrsql_dataset([example], [ABSTAIN_TOKEN], db_path)
        outcome = summary["outcomes"][0]
        assert outcome["gold_failed"]
        assert outcome["error_type"] == "no_such_column"
        assert not outcome["unnecessary_abstention"]
        assert summary["n_gold_failed"] == 1


def test_known_gold_failure_is_excluded_consistently() -> None:
    """A preflight-listed ID is excluded without depending on model output."""
    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / "test.sqlite"
        with sqlite3.connect(db_path) as connection:
            connection.execute("CREATE TABLE records (id INTEGER)")
        example = _record("valid")
        summary = score_ehrsql_dataset(
            [example], ["SELECT id FROM records"], db_path, known_gold_failure_ids={"valid-False"}
        )
        assert summary["n_gold_failed"] == 1
        assert summary["n_scored"] == 0
        assert summary["outcomes"][0]["gold_failed"]


def test_ehrsql_output_preserves_only_exact_abstention() -> None:
    """Only an exact cleaned ABSTAIN is treated as the protocol abstention token."""
    assert clean_ehrsql_output("```\nABSTAIN\n```") == ABSTAIN_TOKEN
    assert clean_ehrsql_output("ABSTAIN because unsupported") == "ABSTAIN because unsupported"
