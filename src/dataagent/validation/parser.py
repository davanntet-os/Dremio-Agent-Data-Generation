"""SQL parsing and static validation using sqlglot (Dremio dialect).

Covers validation stages 2-5:
  Stage 2: parse + statement-type allowlist (SELECT / WITH only)
  Stage 4: read-only enforcement at AST level (no DML/DDL/explain/table(func))
  Stage 5: prohibited patterns + resource limits

Security notes: regexes are used only as *belt-and-braces* checks around the
AST analysis; the parse/tree walk is authoritative. Access control is also
enforced by the read-only Dremio account (defense in depth).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

ALLOWED_STATEMENT_TYPES = (exp.Select, exp.Union, exp.Subquery, exp.CTE)

DANGEROUS_FUNCTIONS = {
    "SHUTDOWN",
    "KILL",
    "REFRESH_METADATA",
}

# Regex belt-and-braces (never primary security control)
_FORBIDDEN_TOKENS = re.compile(
    r"\b(grant|revoke|create|drop|alter|truncate|insert|update|delete|merge|commit|rollback|"
    "start_transaction|call|kill|shutdown|refresh_metadata|pragma)\\b",
    re.IGNORECASE,
)
_COMMENTS = re.compile(r"(--[^\n]*\n?)|(/\*.*?\*/)", re.DOTALL)


@dataclass
class ValidationVerdict:
    ok: bool
    reason: str = ""
    statement_type: str = ""
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    ast: object | None = None


class ParserValidator:
    """Parse-time validation. `known_tables`/`known_columns` enable stage 3."""

    def __init__(self, max_rows: int = 100, max_statements: int = 1) -> None:
        self._max_rows = max_rows
        self._max_statements = max_statements

    def parse_and_validate(
        self,
        sql: str,
        known_tables: set[str] | None = None,
        known_columns: dict[str, set[str]] | None = None,
    ) -> ValidationVerdict:
        cleaned = _strip_trailing_semicolon(_strip_comments(sql))
        if _FORBIDDEN_TOKENS.search(cleaned):
            return ValidationVerdict(False, "Forbidden keyword present (regex pre-check)")

        try:
            statements = sqlglot.parse(cleaned, read="dremio")
        except ParseError as exc:
            return ValidationVerdict(False, f"SQL parse error: {exc}")

        if len(statements) > self._max_statements:
            return ValidationVerdict(False, f"{len(statements)} statements; at most {self._max_statements} allowed")

        assert statements is not None
        tree = statements[0]
        if tree is None:
            return ValidationVerdict(False, "Empty statement")

        verdict = self._check_statement_type(tree)
        if not verdict.ok:
            return verdict

        verdict = self._check_functions(tree)
        if not verdict.ok:
            return verdict

        verdict = self._check_expressions(tree)
        if not verdict.ok:
            return verdict

        verdict = self._check_resource_limits(tree)
        if not verdict.ok:
            return verdict

        if known_tables is not None:
            verdict = self._check_references(tree, known_tables, known_columns or {})
            if not verdict.ok:
                return verdict

        top_type = tree.__class__.__name__
        tables = sorted({t.name for t in tree.find_all(exp.Table)})
        columns = sorted({c.name for c in tree.find_all(exp.Column)})
        return ValidationVerdict(True, statement_type=top_type, tables=tables, columns=columns, ast=tree)

    # -- stage 2: statement-type allowlist -----------------------------------

    def _check_statement_type(self, tree: exp.Expression) -> ValidationVerdict:
        if isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
            return ValidationVerdict(True, statement_type=tree.__class__.__name__, ast=tree)
        if isinstance(tree, exp.With):  # WITH ... SELECT
            return ValidationVerdict(True, statement_type="With", ast=tree)
        # DML/DDL/introspection are all rejected here.
        denied = tree.__class__.__name__
        return ValidationVerdict(False, f"Statement type not allowed: {denied}")

    # -- stage 4: read-only expressions ----------------------------------------

    def _check_functions(self, tree: exp.Expression) -> ValidationVerdict:
        for func in tree.find_all(exp.Anonymous):
            name = (func.name or "").upper()
            if name in DANGEROUS_FUNCTIONS:
                return ValidationVerdict(False, f"Harmful function: {name}")
        return ValidationVerdict(True, ast=tree)

    def _check_expressions(self, tree: exp.Expression) -> ValidationVerdict:
        if tree.find(exp.Into):
            return ValidationVerdict(False, "SELECT INTO not allowed")
        if tree.find(exp.Insert, exp.Delete, exp.Update, exp.Merge, exp.Create, exp.Drop,
                     exp.Alter, exp.Grant, exp.Revoke, exp.TruncateTable, exp.Transaction,
                     exp.Command):
            return ValidationVerdict(False, "Write/phase-control statement detected")
        # Cartesian joins (comma-join or JOIN without ON/USING) are suspicious.
        for join in tree.find_all(exp.Join):
            if join.args.get("on") is None and join.args.get("using") is None:
                return ValidationVerdict(False, "Cartesian join without ON/USING condition")
        return ValidationVerdict(True, ast=tree)

    # -- stage 5: resource limits & prohibited patterns -------------------------

    def _check_resource_limits(self, tree: exp.Expression) -> ValidationVerdict:
        for select in tree.find_all(exp.Select):
            limit = select.args.get("limit")
            if limit is None:
                continue
            value = limit.expression
            try:
                rows = int(value.name)  # type: ignore[union-attr]
            except (AttributeError, ValueError):
                return ValidationVerdict(False, "LIMIT must be a plain integer")
            if rows > self._max_rows:
                return ValidationVerdict(False, f"LIMIT {rows} exceeds max rows {self._max_rows}")
            if rows <= 0:
                return ValidationVerdict(False, "LIMIT must be positive")
        # Bare star *projections* require an explicit LIMIT (COUNT(*) is fine).
        for select in tree.find_all(exp.Select):
            plain_star = any(isinstance(p, exp.Star) for p in select.expressions)
            aliased_star = any(isinstance(a.args.get("this"), exp.Star)
                               for a in select.expressions if isinstance(a, exp.Alias))
            qual_star = any(
                isinstance(p, exp.Column) and isinstance(p.this, exp.Star)
                for p in select.expressions
            )
            if plain_star or aliased_star or qual_star:
                limit = select.args.get("limit")
                if limit is None:
                    return ValidationVerdict(False, "SELECT * requires an explicit LIMIT <= max rows")
        return ValidationVerdict(True, ast=tree)

    # -- stage 3 helper: reference checks ----------------------------------------

    def _check_references(
        self,
        tree: exp.Expression,
        known_tables: set[str],
        known_columns: dict[str, set[str]],
    ) -> ValidationVerdict:
        # CTE names are defined within the statement — not external tables.
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        for table in tree.find_all(exp.Table):
            if table.name.lower() in cte_names:
                continue
            candidates = {table.name.lower()}
            if table.db:
                candidates.add(f"{table.db.lower()}.{table.name.lower()}")
            if table.catalog:
                candidates.add(f"{table.catalog.lower()}.{table.db.lower()}.{table.name.lower()}")
            if candidates.isdisjoint(known_tables):
                return ValidationVerdict(False, f"Unknown table: {table.sql(dialect='dremio')}")
        spec = known_columns or {}
        # columns that are output aliases of the same statement are legitimate
        alias_outputs = {a.output_name.lower() for a in tree.find_all(exp.Alias)}
        for column in tree.find_all(exp.Column):
            if column.name is None or not column.name:
                return ValidationVerdict(False, "Column reference without a name")
            candidates = {column.name.lower()}
            if column.table:
                candidates.add(f"{column.table.lower()}.{column.name.lower()}")
            all_cols: Iterable[str] = set()
            if spec:
                for cols in spec.values():
                    all_cols = set(all_cols) | cols
            allowed = set(all_cols) | alias_outputs
            if candidates.isdisjoint(allowed) and allowed:
                # Unknown *and* unknown-in-any-table: reject as invented column.
                return ValidationVerdict(False, f"Unknown column: {column.name}")
        return ValidationVerdict(True, ast=tree)


def _strip_comments(sql: str) -> str:
    return _COMMENTS.sub(" ", sql)


def _strip_trailing_semicolon(sql: str) -> str:
    return sql.strip().rstrip(";").strip()
