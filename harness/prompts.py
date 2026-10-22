"""Frozen Stage A prompt templates and SQL-output cleanup.

Example:
    prompt = build_prompt_a(schema, question)
"""

from __future__ import annotations

import re

from harness.executor import _split_statements


PROMPT_INSTRUCTION = (
    "You are an expert SQLite developer. Given the database schema and a question, "
    "write one SQLite query that answers the question. Return only the SQL."
)

FIXED_EXAMPLES = """### Examples

#### Example 1
Schema:
CREATE TABLE books (book_id INTEGER PRIMARY KEY, title TEXT, published_year INTEGER);

Question: What are the titles of books published after 2020?

SQL:
SELECT title FROM books WHERE published_year > 2020;

#### Example 2
Schema:
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer TEXT, total REAL);

Question: How many orders does each customer have?

SQL:
SELECT customer, COUNT(*) FROM orders GROUP BY customer;

#### Example 3
Schema:
CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, name TEXT, salary REAL);

Question: What is the highest salary?

SQL:
SELECT MAX(salary) FROM employees;
"""


def build_prompt_a(schema: str, question: str) -> str:
    """Build the frozen plain Stage A prompt exactly around one schema and question."""
    return f"""{PROMPT_INSTRUCTION}

### Schema
{schema}

### Question
{question}

### SQL
"""


def build_prompt_b(schema: str, question: str) -> str:
    """Build the Stage A prompt with its three fixed, non-Spider SQL examples."""
    return f"""{PROMPT_INSTRUCTION}

{FIXED_EXAMPLES}
### Schema
{schema}

### Question
{question}

### SQL
"""


def clean_sql_output(text: str) -> str:
    """Remove Markdown fences, retain the first SQL statement, and trim whitespace."""
    cleaned = re.sub(r"^\s*```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    statements = _split_statements(cleaned)
    return statements[0].strip() if statements else ""
