"""Unit tests for fixed Stage A prompts and SQL output cleanup.

Run with:
    pytest harness/test_prompts.py
"""

from harness.prompts import FIXED_EXAMPLES, build_prompt_a, build_prompt_b, clean_sql_output


def test_prompt_a_has_schema_question_and_sql_marker() -> None:
    """Prompt A preserves the frozen labels and supplied values."""
    prompt = build_prompt_a("CREATE TABLE items (id INTEGER)", "How many items?")
    assert "### Schema\nCREATE TABLE items (id INTEGER)" in prompt
    assert "### Question\nHow many items?" in prompt
    assert prompt.endswith("### SQL\n")


def test_prompt_b_uses_fixed_non_spider_examples() -> None:
    """Prompt B includes each fixed made-up example before the real schema."""
    prompt = build_prompt_b("CREATE TABLE real_table (id INTEGER)", "List ids")
    assert FIXED_EXAMPLES in prompt
    assert prompt.index("CREATE TABLE books") < prompt.index("CREATE TABLE real_table")


def test_clean_sql_output_removes_fence_and_later_statement() -> None:
    """Only the first fenced SQL statement is retained."""
    assert clean_sql_output("```sql\nSELECT 1; SELECT 2;\n```") == "SELECT 1"
