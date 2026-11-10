"""Unit tests for Stage B EHRSQL QLoRA record tokenization."""

from __future__ import annotations

from train.train_ehrsql_qlora import tokenize_record


class FakeTokenizer:
    """A deterministic tokenizer that maps whitespace-separated words to IDs."""

    eos_token = " <eos>"
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> object:
        assert not add_special_tokens
        ids = list(range(1, len(text.split()) + 1))
        if truncation:
            ids = ids[:max_length]
        return type("Tokenized", (), {"input_ids": ids})()


def test_tokenize_record_masks_prompt_and_keeps_sql_completion() -> None:
    """Only completion and EOS tokens contribute to the training loss."""
    record = {"id": "example", "prompt": "one two ", "completion": "three"}
    tokenized = tokenize_record(record, FakeTokenizer(), max_seq_length=10)
    assert tokenized["input_ids"] == [1, 2, 3, 4]
    assert tokenized["labels"] == [-100, -100, 3, 4]
    assert tokenized["attention_mask"] == [1, 1, 1, 1]


def test_tokenize_record_never_labels_truncated_prompt_tokens() -> None:
    """A truncated record does not expose prompt tokens as completion labels."""
    record = {"id": "example", "prompt": "one two three ", "completion": "four"}
    tokenized = tokenize_record(record, FakeTokenizer(), max_seq_length=3)
    assert tokenized["input_ids"] == [1, 2, 3]
    assert tokenized["labels"] == [-100, -100, -100]
