"""Executed-data evidence used by semantic checks.

Builds (and caches) small aggregate queries against the read-only test
environment:
  - distinct status values per status-ish column
  - join-key uniqueness (COUNT(*) vs COUNT(DISTINCT col))
"""

from __future__ import annotations

import re

from dataagent.logging_utils import get_logger
from dataagent.validation.executor import ExecutionService

log = get_logger(__name__)

STATUS_LIKE = re.compile(r"\b(status|state)\b", re.IGNORECASE)


class EvidenceService:
    def __init__(self, executor: ExecutionService) -> None:
        self._executor = executor
        self._distinct_cache: dict[str, dict[str, list[str]]] = {}
        self._uniqueness_cache: dict[str, bool] = {}

    def distinct_status_values(self, fingerprint: str, table_path: str,
                               columns: list[dict]) -> dict[str, list[str]]:
        """{status_col: [values]} from actual data; cached per dataset."""
        if fingerprint in self._distinct_cache:
            return self._distinct_cache[fingerprint]
        out: dict[str, list[str]] = {}
        for col in columns:
            if STATUS_LIKE.search(col["name"]):
                sql = (f'SELECT DISTINCT "{col["name"]}" AS v'
                       f' FROM {table_path} LIMIT 100')
                outcome = self._executor.execute(sql, example_id="evidence:distinct")
                if outcome.status == "completed":
                    out[col["name"].lower()] = sorted(
                        str(r["v"]) for r in outcome.rows if r.get("v") is not None)
        self._distinct_cache[fingerprint] = out
        return out

    def column_uniqueness(self, fingerprint: str, table_path: str,
                          column: str) -> bool:
        """True/False when evidence exists; cached per (table, column)."""
        key = f"{fingerprint}.{column.lower()}"
        if key in self._uniqueness_cache:
            return self._uniqueness_cache[key]
        sql = (f'SELECT COUNT(*) AS n, COUNT(DISTINCT "{column}") AS d'
               f' FROM {table_path}')
        outcome = self._executor.execute(sql, example_id="evidence:uniqueness")
        unique = None
        if outcome.status == "completed" and outcome.rows:
            row = outcome.rows[0]
            unique = row.get("n") == row.get("d")
        self._uniqueness_cache[key] = unique
        return unique

    def date_column_range(self, fingerprint: str, table_path: str,
                          column: str) -> tuple[str, str] | None:
        """(min, max) year strings of a date/timestamp column; cached."""
        if not hasattr(self, "_range_cache"):
            self._range_cache: dict[str, tuple[str, str] | None] = {}
        key = f"{fingerprint}.{column.lower()}"
        if key in self._range_cache:
            return self._range_cache[key]
        sql = (f'SELECT MIN("{column}") AS mn, MAX("{column}") AS mx'
               f' FROM {table_path}')
        outcome = self._executor.execute(sql, example_id="evidence:range")
        rng = None
        if outcome.status == "completed" and outcome.rows:
            mn, mx = outcome.rows[0].get("mn"), outcome.rows[0].get("mx")
            if mn is not None and mx is not None:
                rng = (str(mn)[:4], str(mx)[:4])  # year part
        self._range_cache[key] = rng
        return rng

    def integer_column_range(self, fingerprint: str, table_path: str,
                             column: str) -> tuple[float, float] | None:
        """(min, max) numeric range of a column; cached."""
        if not hasattr(self, "_int_range_cache"):
            self._int_range_cache: dict[str, tuple[float, float] | None] = {}
        key = f"{fingerprint}.{column.lower()}"
        if key in self._int_range_cache:
            return self._int_range_cache[key]
        sql = (f'SELECT MIN("{column}") AS mn, MAX("{column}") AS mx'
               f' FROM {table_path}')
        outcome = self._executor.execute(sql, example_id="evidence:int_range")
        rng = None
        if outcome.status == "completed" and outcome.rows:
            mn, mx = outcome.rows[0].get("mn"), outcome.rows[0].get("mx")
            try:
                if mn is not None and mx is not None:
                    rng = (float(mn), float(mx))
            except (TypeError, ValueError):
                rng = None
        self._int_range_cache[key] = rng
        return rng
