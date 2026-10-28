"""Stage B EHRSQL prompt candidates before one is frozen on validation results."""

from __future__ import annotations


PROMPT_A_INSTRUCTION = """You are an expert SQLite developer working with a de-identified electronic health-record database.
Given the database schema and a question, return exactly one SQLite SELECT or WITH query that answers the question.
If the question cannot be answered from the database, return exactly ABSTAIN.
Return only SQL or ABSTAIN. Do not provide medical advice."""

PROMPT_B_INSTRUCTION = """You are an expert SQLite developer working with a de-identified electronic health-record database.
Use only the tables and columns in the supplied schema. Return exactly one SQLite SELECT or WITH query when the question can be answered from the database.
If the requested information is not represented in the schema or database, return exactly ABSTAIN.
Return only SQL or ABSTAIN. Do not explain the answer or provide medical advice."""


def _build_prompt(instruction: str, schema: str, question: str) -> str:
    """Build one general EHRSQL prompt without examples or clinical guidance."""
    return f"""{instruction}

### Schema
{schema}

### Question
{question}

### SQL or ABSTAIN
"""


def build_ehrsql_prompt_a(schema: str, question: str) -> str:
    """Build the concise Stage B EHRSQL prompt candidate A."""
    return _build_prompt(PROMPT_A_INSTRUCTION, schema, question)


def build_ehrsql_prompt_b(schema: str, question: str) -> str:
    """Build the schema-grounded Stage B EHRSQL prompt candidate B."""
    return _build_prompt(PROMPT_B_INSTRUCTION, schema, question)
