"""Unit tests for Stage B EHRSQL SFT-record construction."""

from __future__ import annotations

import pytest

from train.make_ehrsql_sft_data import build_record, token_length


class FakeTokenizer:
    """Whitespace tokenizer used to test length accounting without downloads."""

    eos_token = " <eos>"

    def __call__(self, text: str, *, add_special_tokens: bool) -> object:
        assert not add_special_tokens
        return type("Tokenized", (), {"input_ids": text.split()})()


def _example(*, impossible: bool = False) -> dict[str, object]:
    """Return one minimal EHRSQL-style record."""
    return {
        "id": "example-id",
        "question": "How many admissions were recorded?",
        "query": "SELECT COUNT(*) FROM admissions",
        "is_impossible": impossible,
    }


def test_build_record_uses_frozen_ehrsql_prompt_b() -> None:
    """SFT records use Prompt B and retain the corresponding SQL completion."""
    record = build_record(_example(), "CREATE TABLE admissions (id INTEGER)")
    assert record["id"] == "example-id"
    assert "Use only the tables and columns" in record["prompt"]
    assert record["completion"] == "SELECT COUNT(*) FROM admissions"


def test_build_record_rejects_impossible_examples() -> None:
    """Impossible examples never enter the answerable-only SFT corpus."""
    with pytest.raises(ValueError, match="must not enter SFT"):
        build_record(_example(impossible=True), "CREATE TABLE admissions (id INTEGER)")


def test_token_length_includes_completion_and_eos() -> None:
    """Length accounting matches raw prompt, SQL completion, and EOS concatenation."""
    record = {"id": "id", "prompt": "one two ", "completion": "three"}
    assert token_length(record, FakeTokenizer()) == 4
