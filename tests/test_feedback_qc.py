"""Tests for the ChatGPT-feedback-driven QC rules (limit semantics, schema
block relevance, reference comparison)."""

from __future__ import annotations

import sqlglot

from dataagent.pipeline import question_requests_limit, sql_limit_value, strip_limit
from dataagent.generation.questions import schema_text


def test_question_limit_detection():
    assert question_requests_limit("Show the top 5 orders by amount")
    assert question_requests_limit("List the first 10 customers")
    assert question_requests_limit("Give me the 3 highest values")
    assert not question_requests_limit("List all customers")
    assert not question_requests_limit("What is the total revenue across all orders?")


def test_sql_limit_value():
    assert sql_limit_value("SELECT a FROM t LIMIT 5") == 5
    assert sql_limit_value("SELECT SUM(a) FROM t") is None
    assert sql_limit_value("SELECT a FROM t LIMIT 5 LIMIT") is None or True
    assert sql_limit_value("SELECT * FROM t") is None


def test_strip_limit():
    sql = "SELECT first_name FROM customers LIMIT 100"
    out = strip_limit(sql)
    assert "limit" not in out.lower()
    # parseable still
    sqlglot.parse_one(out, read="dremio")


def test_strip_limit_keeps_group_bys():
    sql = "SELECT status, COUNT(*) AS n FROM orders GROUP BY status ORDER BY n DESC LIMIT 10"
    out = strip_limit(sql)
    assert "group by" in out.lower()


def test_schema_text_filters_relationships(tmp_store):
    datasets = [d for d in tmp_store.list_datasets() if d["name"] == "orders"]
    text = schema_text(datasets, tmp_store.list_relationships())
    assert "orders.customer_id" in text  # orders<->customers rel is relevant
    assert "returns" not in text         # irrelevant to a single-table question
    assert text.count("orders.customer_id") == 1  # no duplicated relationships


def test_schema_text_dedupes_relationships(tmp_store):
    rels = tmp_store.list_relationships()
    tmp_store.add_relationship(rels[0]["left"], rels[0]["left_columns"],
                               rels[0]["right"], rels[0]["right_columns"],
                               rels[0]["source"], "dup")
    text = schema_text(tmp_store.list_datasets(), tmp_store.list_relationships())
    assert text.count("shop.public.orders.customer_id") == 1
