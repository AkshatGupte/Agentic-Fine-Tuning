"""Unit tests for the Stage A SQLite execution-accuracy scorer.

Run with either:
    python -m unittest harness.test_scorer
    pytest harness/test_scorer.py
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from harness.executor import run_sql
from harness.scorer import compare_results, score_example


class ScorerTests(unittest.TestCase):
    """Test frozen metric rules against a small temporary SQLite database."""

    def setUp(self) -> None:
        """Create a database with predictable rows for every test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.sqlite"
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("CREATE TABLE records (id INTEGER, value REAL)")
            connection.executemany(
                "INSERT INTO records VALUES (?, ?)", [(1, 2.0), (2, 1.0)]
            )

    def tearDown(self) -> None:
        """Remove the temporary test database."""
        self.temp_dir.cleanup()

    def test_gold_sql_as_prediction_is_correct(self) -> None:
        """Using the gold query as prediction must have perfect execution accuracy."""
        sql = "SELECT id, value FROM records ORDER BY id"
        outcome = score_example(self.db_path, sql, sql)
        self.assertTrue(outcome["correct"])
        self.assertTrue(outcome["valid"])

    def test_unordered_rows_are_correct(self) -> None:
        """Rows can differ in order when the gold query has no ORDER BY."""
        outcome = score_example(
            self.db_path,
            "SELECT id FROM records",
            "SELECT id FROM records ORDER BY id DESC",
        )
        self.assertTrue(outcome["correct"])

    def test_ordered_rows_are_wrong_when_order_differs(self) -> None:
        """Rows must match order when gold SQL has ORDER BY."""
        outcome = score_example(
            self.db_path,
            "SELECT id FROM records ORDER BY id ASC",
            "SELECT id FROM records ORDER BY id DESC",
        )
        self.assertFalse(outcome["correct"])
        self.assertEqual(outcome["error_type"], "wrong_result")

    def test_syntax_error_is_invalid(self) -> None:
        """Malformed SQL is reported as an invalid syntax error."""
        outcome = score_example(self.db_path, "SELECT id FROM records", "SELECT FROM records")
        self.assertFalse(outcome["valid"])
        self.assertEqual(outcome["error_type"], "syntax_error")

    def test_missing_table_is_categorized(self) -> None:
        """SQLite missing-table errors get the frozen error type."""
        outcome = score_example(self.db_path, "SELECT id FROM records", "SELECT * FROM absent")
        self.assertFalse(outcome["valid"])
        self.assertEqual(outcome["error_type"], "no_such_table")

    def test_missing_column_is_categorized(self) -> None:
        """SQLite missing-column errors get the frozen error type."""
        outcome = score_example(self.db_path, "SELECT id FROM records", "SELECT absent FROM records")
        self.assertFalse(outcome["valid"])
        self.assertEqual(outcome["error_type"], "no_such_column")

    def test_long_running_query_times_out(self) -> None:
        """The SQLite progress handler interrupts a query that exceeds its limit."""
        query = """
            WITH RECURSIVE counter(value) AS (
                SELECT 1
                UNION ALL
                SELECT value + 1 FROM counter
            )
            SELECT sum(value) FROM counter
        """
        result = run_sql(self.db_path, query, timeout_s=0.01)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "timeout")

    def test_non_select_is_rejected_without_execution(self) -> None:
        """Destructive SQL is rejected before it can reach SQLite."""
        result = run_sql(self.db_path, "DROP TABLE records", timeout_s=10)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "other")
        self.assertTrue(run_sql(self.db_path, "SELECT * FROM records", timeout_s=10).ok)

    def test_float_rounding_and_numeric_equality(self) -> None:
        """The scorer rounds floats and treats integer and float equivalents alike."""
        self.assertTrue(compare_results([(0.1 + 0.2,), (1,)], [(0.3,), (1.0,)], False))


if __name__ == "__main__":
    unittest.main()
