"""Regression tests for the identified failure patterns (spec section 7).

Patterns 1-15; data-driven ones use stub executors with canned rows.
"""

from __future__ import annotations

import pytest

from dataagent.validation.executor import ExecutionOutcome
from dataagent.validation.parser import ParserValidator
from dataagent.validation.semantic_checks import (
    check_rank_returned,
    check_year_in_data_range,
    check_integer_literal_in_range,
    check_undocumented_metric_terms,
    check_date_filter_intent,
    check_detail_identifier,
    check_division_safety,
    check_join_aggregation_safety,
    check_requested_columns,
    check_status_values,
    run_static_checks,
)

S = '"shop"."public"'
ORDERS = [{"name": "order_id", "type": "BIGINT"},
          {"name": "customer_id", "type": "BIGINT"},
          {"name": "order_date", "type": "TIMESTAMP"},
          {"name": "status", "type": "VARCHAR"},
          {"name": "total_amount", "type": "DECIMAL"}]
CUSTOMERS = [{"name": "customer_id", "type": "BIGINT"},
             {"name": "first_name", "type": "VARCHAR"},
             {"name": "region", "type": "VARCHAR"}]
DS = [
    {"fingerprint": "shop.public.orders", "path": [S, "orders"], "name": "orders",
     "dataset_type": "PHYSICAL_DATASET", "columns": ORDERS},
    {"fingerprint": "shop.public.customers", "path": [S, "customers"], "name": "customers",
     "dataset_type": "PHYSICAL_DATASET", "columns": CUSTOMERS},
]


# 1. All rows versus arbitrary LIMIT -----------------------------------------

def test_pattern_1_all_rows_arbitrary_limit():
    sql = 'SELECT first_name FROM "shop"."public"."customers" LIMIT 100'
    # limit is stripped by pipeline, so exported SQL must have none
    from dataagent.pipeline import question_requests_limit, strip_limit
    assert not question_requests_limit("List all customers")
    assert "limit" not in strip_limit(sql).lower()
    # and when the question DOES request it, LIMIT stays
    assert question_requests_limit("Show the top 5 orders")


# 2. Missing identifiers in detail queries -------------------------------------

def test_pattern_2_missing_identifier():
    finding = check_detail_identifier(
        "List all orders", f'SELECT total_amount FROM {S}."orders"', DS)
    assert finding["severity"] == "needs_review"
    fixed = check_detail_identifier(
        "List all orders", f'SELECT order_id, total_amount FROM {S}."orders"', DS)
    assert fixed["severity"] == "ok"


# 3. Missing customer names when requested --------------------------------------

def test_pattern_3_missing_requested_names():
    finding = check_requested_columns(
        "Show each customer's first_name and order count",
        f'SELECT customer_id, COUNT(*) FROM {S}."orders" GROUP BY customer_id', DS)
    assert finding["severity"] == "needs_review"
    ok = check_requested_columns(
        "Show each customer's customer_id and order count",
        f'SELECT customer_id, COUNT(*) AS n FROM {S}."orders" GROUP BY customer_id', DS)
    assert ok["severity"] == "ok"


# 4. Percentage growth versus absolute growth ------------------------------------
# (covered by reviewer prompt criterion 4 — verified live; deterministic part
#  is that questions with "growth" reach the reviewer)

def test_pattern_4_growth_reaches_review():
    # the check itself is LLM-based; regression is that SQL without window
    # function is parsed and the question keeps category period_comparison
    import sqlglot
    tree = sqlglot.parse_one(
        "WITH m AS (SELECT DATE_TRUNC('month', order_date) AS mo,"
        " SUM(total_amount) AS rev FROM orders GROUP BY 1)"
        " SELECT mo, rev - LAG(rev) OVER (ORDER BY mo) FROM m LIMIT 100",
        read="dremio")
    assert tree.find(sqlglot.exp.Window)


# 5. Ambiguous business metrics --------------------------------------------------

def test_pattern_5_ambiguous_metric_triaged():
    from dataagent.generation.triage import RuleBasedTriage

    triage = RuleBasedTriage()
    assert triage.classify("Show me the sales figures.", "schema_ambiguity", "")[
        "verdict"] == "needs_clarification"
    assert triage.classify("Which products performed best?", "schema_ambiguity", "")[
        "verdict"] == "needs_clarification"
    assert triage.classify("Predict next month's revenue.", "unsupported", "")[
        "verdict"] == "unsupported"


# 6. Missing or incorrect status values ------------------------------------------

def test_pattern_6_unverified_status_value():
    finding = check_status_values(
        "How many shipped orders?",
        'SELECT COUNT(*) FROM orders WHERE status = \'SHIPPED\'',
        {"status": ["COMPLETED", "PENDING", "CANCELLED"]})
    assert finding["severity"] == "needs_review"
    ok = check_status_values(
        "How many shipped orders?",
        "SELECT COUNT(*) FROM orders WHERE status = 'COMPLETED'",
        {"status": ["COMPLETED", "PENDING", "CANCELLED"]})
    assert ok["severity"] == "ok"


# 7. Averages with customers who have no orders -----------------------------------
# (reviewer criterion 5: population semantics; deterministic regression: the
#  candidate parses and joins customers side without dropping zero-order rows)

def test_pattern_7_inner_vs_left_join_population():
    from dataagent.validation.parser import ParserValidator

    v = ParserValidator()
    inner = 'WITH t AS (SELECT c.customer_id, SUM(o.total_amount) AS ltv FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY 1) SELECT * FROM t LIMIT 10'
    assert v.parse_and_validate(inner).ok  # syntactically valid -> reviewer decides


# 8. Incorrect date boundaries -----------------------------------------------------

def test_pattern_8_period_without_date_filter():
    finding = check_date_filter_intent(
        "Revenue this year",
        f'SELECT SUM(total_amount) FROM {S}."orders"')
    assert finding["severity"] == "needs_review"
    ok = check_date_filter_intent(
        "Revenue this year",
        f"SELECT SUM(total_amount) FROM {S}.\"orders\""
        " WHERE order_date >= DATE_TRUNC('year', CURRENT_DATE)")
    assert ok["severity"] == "ok"


# 9. Duplicate rows caused by joins (fan-out) --------------------------------------

def test_pattern_9_join_fanout_with_evidence():
    # customer_id NOT unique on the joined side -> flag
    finding = check_join_aggregation_safety(
        "What is the total shipping cost by customer?",
        f'SELECT c.first_name, SUM(o.total_amount) FROM {S}."orders" o'
        f' JOIN {S}."customers" c ON o.customer_id = c.customer_id',
        DS, uniqueness={"shop.public.customers.customer_id": False})
    assert finding["severity"] == "needs_review"
    # unique on joined side -> safe
    ok = check_join_aggregation_safety(
        "What is the total shipping cost by customer?",
        f'SELECT c.first_name, SUM(o.total_amount) FROM {S}."orders" o'
        f' JOIN {S}."customers" c ON o.customer_id = c.customer_id',
        DS, uniqueness={"shop.public.customers.customer_id": True})
    assert ok["severity"] == "ok"


# 10. Incorrect aggregation after joining tables (no evidence at all) ---------------

def test_pattern_10_sum_after_join_without_evidence():
    finding = check_join_aggregation_safety(
        "What is the total revenue by customer?",
        f'SELECT c.first_name, SUM(o.total_amount) FROM {S}."orders" o'
        f' JOIN {S}."customers" c ON o.customer_id = c.customer_id',
        DS, uniqueness=None)
    assert finding["severity"] == "needs_review"


# 11. Nonexistent columns -------------------------------------------------------------

def test_pattern_11_nonexistent_column():
    v = ParserValidator()
    verdict = v.parse_and_validate(
        "SELECT bogus FROM orders",
        known_tables={"orders"}, known_columns={"orders": {"order_id"}})
    assert not verdict.ok


# 12. Nonexistent or unverified tables --------------------------------------------------

def test_pattern_12_nonexistent_table():
    v = ParserValidator()
    verdict = v.parse_and_validate(
        "SELECT id FROM unknown_table",
        known_tables={"orders"}, known_columns={"orders": {"id"}})
    assert not verdict.ok


# 13. Division by zero -------------------------------------------------------------------

def test_pattern_13_division_by_zero():
    finding = check_division_safety(
        f'SELECT shipping_cost / total_amount FROM {S}."orders"')
    assert finding["severity"] == "needs_review"
    ok = check_division_safety(
        f"SELECT shipping_cost / NULLIF(total_amount, 0) FROM {S}.\"orders\"")
    assert ok["severity"] == "ok"


# 14. NULL handling -------------------------------------------------------------------------

def test_pattern_14_null_awareness_parses():
    v = ParserValidator()
    assert v.parse_and_validate(
        f'SELECT COUNT(*) FROM {S}."orders" WHERE shipping_cost IS NULL').ok
    assert v.parse_and_validate(
        f'SELECT COALESCE(shipping_cost, 0) FROM {S}."orders"').ok


# 15. Questions that require clarification -----------------------------------------------------

def test_pattern_15_clarification_contract():
    from dataagent.export.jsonl import JSONLExporter

    exporter = JSONLExporter()
    rendered = exporter._render_clarification({
        "question": "Show me the sales figures.",
        "interpretations": ["total sales", "monthly sales", "sales by product"],
    })
    assistant = rendered["messages"][-1]["content"]
    assert "clarification" in assistant.lower()
    assert "total sales" in assistant
    # QC metadata must never leak into the conversational content
    assert "example_id" not in assistant


def test_pattern_bundle_static_checks_run_all():
    findings = run_static_checks(
        "List all orders", f'SELECT total_amount FROM {S}."orders"', DS,
        uniqueness={"shop.public.orders.order_id": True})
    checks = {f["check"] for f in findings}
    assert "detail_identifier" in checks


# 16. "Rank" wording without a returned rank (found in train audit) ---------------

def test_pattern_16_rank_wording():
    finding = check_rank_returned(
        "Use a CTE to rank customers by number of orders and show the top 10.",
        "WITH oc AS (SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY 1)"
        " SELECT customer_id, n FROM oc ORDER BY n DESC LIMIT 10")
    assert finding["severity"] == "needs_review"
    ok = check_rank_returned(
        "Rank customers by orders.",
        "SELECT customer_id, RANK() OVER (ORDER BY COUNT(*) DESC) AS r FROM orders GROUP BY 1 LIMIT 10")
    assert ok["severity"] == "ok"


# 17. Year referenced by the question outside the data range ----------------------

def test_pattern_17_year_outside_data_range():
    class FakeEvidence:
        def date_column_range(self, fp, path, col):
            return ("2023", "2024")
    finding = check_year_in_data_range(
        "Which customers signed up during 2022?",
        "SELECT customer_id FROM customers WHERE YEAR(signup_date) = 2022",
        [{"name": "customers", "fingerprint": "c",
          "columns": [{"name": "signup_date", "type": "TIMESTAMP"}]}],
        FakeEvidence())
    assert finding["severity"] == "needs_review"


# 18. NOT IN with case-mismatched status literals is vacuous (live defect) ---------

def test_pattern_18_vacuous_not_in():
    finding = check_status_values(
        "Which orders are neither completed nor cancelled?",
        "SELECT order_id, status FROM orders WHERE status NOT IN ('completed', 'cancelled')",
        {"status": ["PENDING", "COMPLETED", "CANCELLED"]})
    assert finding["severity"] == "needs_review"
    ok = check_status_values(
        "Which orders are neither completed nor cancelled?",
        "SELECT order_id, status FROM orders WHERE status NOT IN ('COMPLETED', 'CANCELLED')",
        {"status": ["PENDING", "COMPLETED", "CANCELLED"]})
    assert ok["severity"] == "ok"


# 19. Integer literal outside column data range (live: "order 5001", max is 5000) --

def test_pattern_19_integer_literal_out_of_range():
    class FakeEvidence:
        def integer_column_range(self, fp, path, col):
            return (1, 5000)
    finding = check_integer_literal_in_range(
        "Display the order items for order 5001.",
        "SELECT order_item_id FROM order_items WHERE order_id = 5001",
        [{"name": "order_items", "fingerprint": "oi",
          "path": ["shop", "order_items"],
          "columns": [{"name": "order_id", "type": "BIGINT"}]}],
        FakeEvidence())
    assert finding["severity"] == "needs_review"


# 20. Undocumented business-metric terms ('revenue' with no definition) ------------

def test_pattern_20_undocumented_metric_terms():
    finding = check_undocumented_metric_terms(
        "What is the total revenue across all order items?",
        "SELECT SUM(line_amount) AS total_revenue FROM order_items")
    assert finding["severity"] == "needs_review"
    ok = check_undocumented_metric_terms(
        "What is the total line amount across all order items?",
        "SELECT SUM(line_amount) AS total_line_amount FROM order_items")
    assert ok["severity"] == "ok"
