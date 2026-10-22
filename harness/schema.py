"""Read SQLite schemas for Stage A prompts.

Example:
    schema = get_schema(Path("data/raw/spider/database/academic/academic.sqlite"))
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def get_schema(db_path: Path) -> str:
    """Return ``CREATE TABLE`` statements in the database's original order."""
    database_uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND sql IS NOT NULL
            ORDER BY rowid
            """
        ).fetchall()
    return "\n\n".join(row[0] for row in rows)
