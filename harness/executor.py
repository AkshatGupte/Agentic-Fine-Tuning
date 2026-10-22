"""Safely execute one read-only SQLite query with a timeout.

Example:
    result = run_sql(Path("database.sqlite"), "SELECT * FROM users", timeout_s=10)
"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
import time
from pathlib import Path


@dataclass(frozen=True)
class Result:
    """The outcome of executing one SQL query."""

    ok: bool
    rows: list[tuple[object, ...]] | None
    error_type: str | None
    error_message: str | None


def _split_statements(sql: str) -> list[str]:
    """Split SQL on semicolons outside strings, quoted identifiers, and comments."""
    statements: list[str] = []
    start = 0
    index = 0
    state = "normal"
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "normal":
            if char == "'":
                state = "single_quote"
            elif char == '"':
                state = "double_quote"
            elif char == "`":
                state = "backtick"
            elif char == "[":
                state = "bracket"
            elif char == "-" and next_char == "-":
                state = "line_comment"
                index += 1
            elif char == "/" and next_char == "*":
                state = "block_comment"
                index += 1
            elif char == ";":
                statement = sql[start:index].strip()
                if statement:
                    statements.append(statement)
                start = index + 1
        elif state == "single_quote":
            if char == "'":
                if next_char == "'":
                    index += 1
                else:
                    state = "normal"
        elif state == "double_quote":
            if char == '"':
                if next_char == '"':
                    index += 1
                else:
                    state = "normal"
        elif state == "backtick" and char == "`":
            state = "normal"
        elif state == "bracket" and char == "]":
            state = "normal"
        elif state == "line_comment" and char in "\r\n":
            state = "normal"
        elif state == "block_comment" and char == "*" and next_char == "/":
            state = "normal"
            index += 1
        index += 1

    statement = sql[start:].strip()
    if statement:
        statements.append(statement)
    return statements


def _without_leading_comments(statement: str) -> str:
    """Remove leading whitespace and SQLite comments before checking its keyword."""
    remaining = statement.lstrip()
    while True:
        if remaining.startswith("--"):
            newline = remaining.find("\n")
            remaining = "" if newline == -1 else remaining[newline + 1 :].lstrip()
        elif remaining.startswith("/*"):
            closing = remaining.find("*/", 2)
            if closing == -1:
                return remaining
            remaining = remaining[closing + 2 :].lstrip()
        else:
            return remaining


def _is_allowed_query(statement: str) -> bool:
    """Return whether a statement begins with the allowed SELECT or WITH keyword."""
    keyword = _without_leading_comments(statement).split(maxsplit=1)[0:1]
    return bool(keyword) and keyword[0].upper() in {"SELECT", "WITH"}


def _error_type(error: sqlite3.Error, timed_out: bool) -> str:
    """Map SQLite errors to the frozen Stage A error categories."""
    if timed_out:
        return "timeout"
    message = str(error).lower()
    if "no such table" in message:
        return "no_such_table"
    if "no such column" in message:
        return "no_such_column"
    if any(marker in message for marker in ("syntax error", "incomplete input", "unrecognized token")):
        return "syntax_error"
    return "other"


def run_sql(db_path: Path, sql: str, timeout_s: float) -> Result:
    """Run one SELECT/WITH query read-only, returning rows or a categorized failure."""
    if not sql or not sql.strip():
        return Result(False, None, "empty_prediction", "SQL is empty.")
    statements = _split_statements(sql)
    if len(statements) != 1 or not _is_allowed_query(statements[0]):
        return Result(
            False,
            None,
            "other",
            "Only one SELECT or read-only WITH query is allowed.",
        )

    timed_out = False
    started_at = time.monotonic()

    def check_timeout() -> int:
        nonlocal timed_out
        if time.monotonic() - started_at > timeout_s:
            timed_out = True
            return 1
        return 0

    database_uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    try:
        with sqlite3.connect(database_uri, uri=True) as connection:
            connection.set_progress_handler(check_timeout, 1_000)
            try:
                rows = connection.execute(statements[0]).fetchall()
            finally:
                connection.set_progress_handler(None, 0)
    except sqlite3.Error as error:
        return Result(False, None, _error_type(error, timed_out), str(error))
    return Result(True, rows, None, None)
