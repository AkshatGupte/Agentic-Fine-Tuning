"""Clean Stage B EHRSQL model output while preserving the ABSTAIN contract."""

from __future__ import annotations

import re

from harness.ehrsql_scorer import ABSTAIN_TOKEN
from harness.prompts import clean_sql_output


def clean_ehrsql_output(text: str) -> str:
    """Return exact ABSTAIN or the first cleaned SQL statement from model output."""
    cleaned = re.sub(r"^\s*```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    if cleaned == ABSTAIN_TOKEN:
        return ABSTAIN_TOKEN
    return clean_sql_output(cleaned)
