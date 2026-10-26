"""Reusable semantic checks between question text and candidate SQL.

Deterministic, question-agnostic checks that run before/after execution:

  - detail_identifier     : detail listings must include an identifying column
  - requested_columns     : named fields the question mentions must appear
  - status_values         : status literals in SQL must exist in actual data
  - date_filter_intent    : period words ("this year"...) require a date WHERE
  - division_safety       : raw division must guard zero/NULL denominators
  - join_aggregation_safety : SUM/AVG after joins must not fan out rows
    (verified with executed COUNT(*) vs COUNT(DISTINCT) evidence)

Severity levels: "ok" | "needs_review" | "reject".
"""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp as E

_ID_SUFFIX = re.compile(r"(_id|_no|_number|_code)$", re.IGNORECASE)
_PERIOD_WORDS = re.compile(
    r"\b(this|last|current|previous|past)\s+(year|month|week|quarter|day|"
    r"fiscal\s+year)\b|\byear\s+to\s+date\b|\bmtd\b|\bqtd\b|\bytd\b",
    re.IGNORECASE)


def _parse(sql: str) -> E.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="dremio")
    except Exception:
        return None


def _top_select(tree: E.Expression) -> E.Select | None:
    if isinstance(tree, E.Select):
        return tree
    sel = tree.find(E.Select)
    return sel


def check_detail_identifier(question: str, sql: str, datasets: list[dict]) -> dict:
    """Detail queries (no aggregation/grouping, plural intent) must identify rows."""
    tree = _parse(sql)
    if tree is None:
        return {"check": "detail_identifier", "severity": "ok", "reason": ""}
    top = _top_select(tree)
    if top is None:
        return {"check": "detail_identifier", "severity": "ok", "reason": ""}
    has_aggregate = top.find(E.AggFunc) is not None
    has_group = top.args.get("group") is not None
    if has_aggregate or has_group:
        return {"check": "detail_identifier", "severity": "ok", "reason": ""}
    if not re.search(r"\b(all|every|each|list|show|display|details?)\b", question, re.I):
        return {"check": "detail_identifier", "severity": "ok", "reason": ""}
    projections = []
    star = False
    for p in top.expressions:
        if isinstance(p, E.Star) or (isinstance(p, E.Column) and isinstance(p.this, E.Star)):
            star = True
        elif isinstance(p, E.Alias):
            projections.append(p.output_name.lower())
        elif isinstance(p, E.Column):
            projections.append(p.name.lower())
        else:
            projections.extend(col.name.lower() for col in p.find_all(E.Column))
    if star:
        return {"check": "detail_identifier", "severity": "ok", "reason": ""}
    id_cols = [c["name"] for d in datasets for c in d["columns"]
               if _ID_SUFFIX.search(c["name"])]
    id_lower = {c.lower() for c in id_cols}
    has_id = any(n in id_lower or n.endswith(tuple(id_lower)) for n in projections)
    if has_id:
        return {"check": "detail_identifier", "severity": "ok", "reason": ""}
    return {"check": "detail_identifier", "severity": "needs_review",
            "reason": "detail listing without an identifying column (e.g. order_id)"}


def check_requested_columns(question: str, sql: str, datasets: list[dict]) -> dict:
    """Columns mentioned by name in the question must appear in the projection."""
    tree = _parse(sql)
    if tree is None:
        return {"check": "requested_columns", "severity": "ok", "reason": ""}
    top = _top_select(tree)
    if top is None:
        return {"check": "requested_columns", "severity": "ok", "reason": ""}
    proj_cols: set[str] = set()
    for p in top.expressions:
        for col in p.find_all(E.Column):
            proj_cols.add(col.name.lower())
        if isinstance(p, E.Alias):
            proj_cols.add(p.output_name.lower())
    for d in datasets:
        for c in d["columns"]:
            name = c["name"]
            # snake_case name appears in question verbatim (underscores optional)
            loose = re.sub(r"[_\s]+", " ", name).lower()
            if loose in re.sub(r"[_]+", " ", question).lower():
                if name.lower() not in proj_cols:
                    return {"check": "requested_columns", "severity": "needs_review",
                            "reason": f"question mentions '{name}' but the projection omits it"}
    return {"check": "requested_columns", "severity": "ok", "reason": ""}


def check_status_values(question: str, sql: str,
                        distinct_values: dict[str, list[str]] | None = None) -> dict:
    """Status literals filtered on must exist in the actual data.

    Handles EQ, NE, IN and NOT IN predicates; comparison is case-insensitive
    so that lowercase literals against uppercase data (which silently make
    NOT IN vacuous) are flagged.
    """
    tree = _parse(sql)
    if tree is None:
        return {"check": "status_values", "severity": "ok", "reason": ""}
    for col in tree.find_all(E.Column):
        if "status" not in col.name.lower():
            continue
        parent = col.parent
        literals: list[str] = []
        if isinstance(parent, (E.EQ, E.NEQ)) and isinstance(parent.expression, E.Literal):
            if parent.expression.is_string:
                literals.append(parent.expression.this)
        elif isinstance(parent, E.In):
            for lit in parent.expressions:
                if isinstance(lit, E.Literal) and lit.is_string:
                    literals.append(lit.this)
        if not literals:
            continue
        # EXACT match against actual data values: sources like Postgres match
        # strings case-sensitively, so 'completed' would make EQ/IN return
        # nothing and NOT IN vacuously match every row.
        known = {str(v) for v in (distinct_values or {}).get(col.name.lower(), [])}
        if not known:
            continue
        for value in literals:
            if str(value) not in known:
                hint = [v for v in known if str(v).upper() == str(value).upper()]
                return {"check": "status_values", "severity": "needs_review",
                        "reason": f"status value '{value}' does not exactly match actual "
                                  f"data values (known: {sorted(known)[:8]}"
                                  + (f"; case-mismatch: did you mean '{hint[0]}'?" if hint else "") + ")"}
    return {"check": "status_values", "severity": "ok", "reason": ""}


def check_date_filter_intent(question: str, sql: str) -> dict:
    """Period words in the question require a date predicate in the SQL."""
    tree = _parse(sql)
    if tree is None:
        return {"check": "date_filter_intent", "severity": "ok", "reason": ""}
    if not _PERIOD_WORDS.search(question):
        return {"check": "date_filter_intent", "severity": "ok", "reason": ""}
    top = _top_select(tree)
    where = top.args.get("where") if top is not None else None
    if where is None:
        return {"check": "date_filter_intent", "severity": "needs_review",
                "reason": "question references a period but SQL has no WHERE date filter"}
    date_cols = [c["name"] for d in [] for c in d.get("columns", [])]
    del date_cols  # datasets not needed here; presence of WHERE is the heuristic
    has_date_ref = any(col.name.lower().find("date") >= 0 or
                       col.name.lower().find("timestamp") >= 0 or
                       col.name.lower().find("time") >= 0
                       for col in where.find_all(E.Column))
    if not has_date_ref:
        return {"check": "date_filter_intent", "severity": "needs_review",
                "reason": "WHERE filter does not reference a date/timestamp column"}
    return {"check": "date_filter_intent", "severity": "ok", "reason": ""}


def check_division_safety(sql: str) -> dict:
    """Raw division should guard zero/NULL denominators (NULLIF or CASE)."""
    tree = _parse(sql)
    if tree is None:
        return {"check": "division_safety", "severity": "ok", "reason": ""}
    for div in tree.find_all(E.Div):
        denom = div.expression
        # denominator guarded by NULLIF(...) or a CASE expression
        if isinstance(denom, (E.Nullif, E.Case)) or denom.find(E.Nullif):
            continue
        # or the whole division is wrapped in a CASE guard
        parent = div.parent
        guarded = False
        while parent is not None:
            if isinstance(parent, E.Case):
                guarded = True
                break
            parent = parent.parent
        if not guarded:
            return {"check": "division_safety", "severity": "needs_review",
                    "reason": "division without NULLIF/CASE guard for zero/NULL denominator"}
    return {"check": "division_safety", "severity": "ok", "reason": ""}


def check_join_aggregation_safety(
        question: str, sql: str, datasets: list[dict],
        uniqueness: dict[str, bool] | None = None) -> dict:
    """Aggregating sums after joins must not fan out rows.

    Evidence-based: the join key on the "dimension" side must be unique
    (executed COUNT(*) = COUNT(DISTINCT) evidence passed in `uniqueness`
    keyed as '<fingerprint>.<col>'). Without evidence, only flag obvious
    many-to-many patterns as needs_review.
    """
    tree = _parse(sql)
    if tree is None:
        return {"check": "join_aggregation_safety", "severity": "ok", "reason": ""}
    top = _top_select(tree)
    if top is None or not top.find(E.Join):
        return {"check": "join_aggregation_safety", "severity": "ok", "reason": ""}
    if not any(a.key in ("sum", "avg") for a in top.find_all(E.AggFunc)):
        return {"check": "join_aggregation_safety", "severity": "ok", "reason": ""}
    # resolve alias -> dataset fingerprint
    fp_by_alias: dict[str, str] = {}
    for tbl in top.find_all(E.Table):
        fp = next((d["fingerprint"] for d in datasets
                   if d["name"].lower() == tbl.name.lower()), None)
        if fp:
            fp_by_alias[tbl.alias_or_name.lower()] = fp
            fp_by_alias[tbl.name.lower()] = fp
    for join in top.find_all(E.Join):
        cond = join.args.get("on")
        if not (isinstance(cond, E.EQ) and isinstance(cond.this, E.Column)
                and isinstance(cond.expression, E.Column)):
            continue
        for key_col, other_col in ((cond.this, cond.expression),
                                   (cond.expression, cond.this)):
            tbl_name = (key_col.table or "").lower()
            fp = fp_by_alias.get(tbl_name)
            if not fp:
                continue
            key = f"{fp}.{key_col.name.lower()}"
            if uniqueness is not None:
                if uniqueness.get(key) is False:
                    return {
                        "check": "join_aggregation_safety",
                        "severity": "needs_review",
                        "reason": f"join key {tbl_name}.{key_col.name} is not unique; "
                                  "SUM/AVG over this join may double-count rows"}
                if uniqueness.get(key) is True:
                    return {"check": "join_aggregation_safety", "severity": "ok", "reason": ""}
    # no evidence available at all: be conservative for metric questions
    if re.search(r"\b(total|revenue|sum|sum of|amount)\b", question, re.I):
        return {"check": "join_aggregation_safety", "severity": "needs_review",
                "reason": "SUM/AVG over a join without uniqueness evidence"}
    return {"check": "join_aggregation_safety", "severity": "ok", "reason": ""}


def check_distinct_count_intent(question: str, sql: str) -> dict:
    """"Unique/distinct X" questions must use COUNT(DISTINCT ...)."""
    tree = _parse(sql)
    if tree is None:
        return {"check": "distinct_count", "severity": "ok", "reason": ""}
    if not re.search(r"\b(unique|distinct)\b\s+\w+", question, re.IGNORECASE):
        return {"check": "distinct_count", "severity": "ok", "reason": ""}
    for count in tree.find_all(E.Count):
        if count.this and isinstance(count.this, E.Distinct):
            return {"check": "distinct_count", "severity": "ok", "reason": ""}
        if count.find(E.Distinct):
            return {"check": "distinct_count", "severity": "ok", "reason": ""}
    return {"check": "distinct_count", "severity": "needs_review",
            "reason": "question asks for unique/distinct items but SQL uses plain COUNT(*)"}


def check_rank_returned(question: str, sql: str) -> dict:
    """Questions about ranking must actually return a rank column."""
    if not re.search(r"\brank\b|\branking\b", question, re.IGNORECASE):
        return {"check": "rank_returned", "severity": "ok", "reason": ""}
    tree = _parse(sql)
    if tree is None:
        return {"check": "rank_returned", "severity": "ok", "reason": ""}
    has_rank_window = any(
        isinstance(w, (E.Rank, E.RowNumber, E.DenseRank))
        for w in tree.find_all(E.Rank, E.RowNumber, E.DenseRank))
    if has_rank_window:
        return {"check": "rank_returned", "severity": "ok", "reason": ""}
    return {"check": "rank_returned", "severity": "needs_review",
            "reason": "question mentions ranking but SQL returns no rank column"}


def check_year_in_data_range(question: str, sql: str, datasets: list[dict],
                             evidence=None) -> dict:
    """Years referenced by the question must exist in the date-column range."""
    years = {int(y) for y in re.findall(r"\b(20\d{2})\b", question)}
    if not years or evidence is None:
        return {"check": "year_in_data_range", "severity": "ok", "reason": ""}
    tree = _parse(sql)
    if tree is None:
        return {"check": "year_in_data_range", "severity": "ok", "reason": ""}
    for ds in datasets:
        for c in ds["columns"]:
            if not re.search(r"(date|timestamp|_time|_at)$", c["name"], re.IGNORECASE):
                continue
            rng = evidence.date_column_range(ds["fingerprint"],
                                             ".".join(ds.get("path", [ds["name"]])), c["name"])
            if rng is None:
                continue
            lo, hi = int(rng[0]), int(rng[1])
            outside = years - set(range(lo, hi + 1))
            if outside:
                return {"check": "year_in_data_range", "severity": "needs_review",
                        "reason": f"year(s) {sorted(outside)} fall outside the data range "
                                  f"of {ds['name']}.{c['name']} ({lo}-{hi})"}
    return {"check": "year_in_data_range", "severity": "ok", "reason": ""}


def check_integer_literal_in_range(question: str, sql: str, datasets: list[dict],
                                   evidence=None) -> dict:
    """Integer literals in predicates must exist within the column's data range.

    Catches questions referencing IDs/values that don't exist in the data
    (e.g. "order 5001" when max order_id is 5000): the query succeeds but
    answers nothing.
    """
    tree = _parse(sql)
    if tree is None or evidence is None:
        return {"check": "integer_literal_in_range", "severity": "ok", "reason": ""}
    fp_by_name = {d["name"].lower(): d["fingerprint"] for d in datasets}
    for col in tree.find_all(E.Column):
        parent = col.parent
        if not isinstance(parent, (E.EQ, E.LT, E.GT, E.LTE, E.GTE)):
            continue
        other = parent.expression if parent.this is col else parent.this
        if not (isinstance(other, E.Literal) and (other.is_int or other.is_number)):
            continue
        try:
            value = float(other.name if other.name is not None else other.this)
        except (TypeError, ValueError):
            continue
        fp = fp_by_name.get((col.table or "").lower())
        if not fp:
            # unqualified column: resolve via column existence (exactly one match)
            matches = [d for d in datasets
                       if any(c["name"].lower() == col.name.lower()
                              for c in d["columns"])]
            if len(matches) != 1:
                continue
            ds = matches[0]
            fp = ds["fingerprint"]
        else:
            ds = next((d for d in datasets if d["fingerprint"] == fp), None)
        table_path = ".".join(ds.get("path", [ds["name"]])) if ds else ""
        rng = evidence.integer_column_range(fp, table_path, col.name)
        if rng is None:
            continue
        if not (rng[0] <= value <= rng[1]):
            return {"check": "integer_literal_in_range", "severity": "needs_review",
                    "reason": f"value {value:g} falls outside the data range of "
                              f"{col.name} ({rng[0]:g}-{rng[1]:g})"}
    return {"check": "integer_literal_in_range", "severity": "ok", "reason": ""}


def check_undocumented_metric_terms(question: str, sql: str) -> dict:
    """'revenue'/'sales'/'profit' in questions need a documented definition.

    total_amount is documented as "order total including taxes"; line_amount
    as "quantity * unit_price". Calling them "revenue" invents a business
    definition, so such examples require review.
    """
    if not re.search(r"\b(revenue|sales|profit|margin|turnover)\b", question, re.I):
        return {"check": "undocumented_metric_terms", "severity": "ok", "reason": ""}
    return {"check": "undocumented_metric_terms", "severity": "needs_review",
            "reason": "question uses a business metric term without a documented "
                      "definition; use schema-anchored terminology (e.g. 'order amount')"}


def run_static_checks(question: str, sql: str, datasets: list[dict],
                      uniqueness: dict[str, bool] | None = None,
                      evidence=None) -> list[dict]:
    findings = [
        check_detail_identifier(question, sql, datasets),
        check_requested_columns(question, sql, datasets),
        check_date_filter_intent(question, sql),
        check_division_safety(sql),
        check_distinct_count_intent(question, sql),
        check_rank_returned(question, sql),
        check_join_aggregation_safety(question, sql, datasets, uniqueness),
        check_integer_literal_in_range(question, sql, datasets, evidence),
        check_undocumented_metric_terms(question, sql),
    ]
    findings.append(check_year_in_data_range(question, sql, datasets, evidence))
    return [f for f in findings if f["severity"] != "ok"]
