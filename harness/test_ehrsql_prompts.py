"""Unit tests for the two unfrozen Stage B EHRSQL prompt candidates."""

from harness.ehrsql_prompts import build_ehrsql_prompt_a, build_ehrsql_prompt_b


def test_ehrsql_prompts_include_schema_question_and_abstain_rule() -> None:
    """Both candidates must state the fixed output contract and supplied values."""
    for builder in (build_ehrsql_prompt_a, build_ehrsql_prompt_b):
        prompt = builder("CREATE TABLE patients (id INTEGER)", "What is patient 1's age?")
        assert "CREATE TABLE patients (id INTEGER)" in prompt
        assert "What is patient 1's age?" in prompt
        assert "exactly ABSTAIN" in prompt
        assert prompt.endswith("### SQL or ABSTAIN\n")


def test_ehrsql_prompt_b_requires_schema_grounding() -> None:
    """Candidate B differs by explicitly restricting output to supplied schema fields."""
    prompt_a = build_ehrsql_prompt_a("CREATE TABLE x (id INTEGER)", "List ids")
    prompt_b = build_ehrsql_prompt_b("CREATE TABLE x (id INTEGER)", "List ids")
    assert "Use only the tables and columns" not in prompt_a
    assert "Use only the tables and columns" in prompt_b
