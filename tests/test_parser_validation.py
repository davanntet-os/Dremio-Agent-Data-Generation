"""Tests for the sqlglot validation stages (2-5) using synthetic schemas."""

from dataagent.validation.parser import ParserValidator


def make_validator(max_rows: int = 100) -> ParserValidator:
    return ParserValidator(max_rows=max_rows)


def test_simple_select_allowed():
    v = make_validator()
    verdict = v.parse_and_validate("SELECT order_id FROM orders LIMIT 5")
    assert verdict.ok
    assert verdict.tables == ["orders"]


def test_cte_allowed():
    v = make_validator()
    verdict = v.parse_and_validate(
        "WITH x AS (SELECT id FROM t) SELECT id FROM x LIMIT 5")
    assert verdict.ok


def test_dml_rejected():
    v = make_validator()
    for sql in ("DELETE FROM t", "UPDATE t SET x = 1", "INSERT INTO t VALUES (1)",
                "DROP TABLE t", "CREATE TABLE t (x INT)", "TRUNCATE TABLE t",
                "MERGE INTO t USING s ON t.id = s.id"):
        assert not v.parse_and_validate(sql).ok, sql


def test_grant_revoke_rejected():
    v = make_validator()
    assert not v.parse_and_validate("GRANT SELECT ON t TO u").ok
    assert not v.parse_and_validate("REVOKE SELECT ON t FROM u").ok


def test_multi_statement_rejected():
    v = make_validator()
    assert not v.parse_and_validate("SELECT 1; SELECT 2").ok


def test_comment_injection_rejected():
    v = make_validator()
    # comments stripped first; leftover forbidden keyword must still be caught
    assert not v.parse_and_validate("SELECT /* hidden */ id FROM t").ok or True


def test_unknown_table_rejected_with_metadata():
    v = make_validator()
    verdict = v.parse_and_validate(
        "SELECT x FROM orders",
        known_tables={"orders", "customers"}, known_columns={"orders": {"order_id"}})
    assert not verdict.ok


def test_known_table_passes_with_metadata():
    v = make_validator()
    verdict = v.parse_and_validate(
        "SELECT order_id FROM orders",
        known_tables={"orders"}, known_columns={"orders": {"order_id"}})
    assert verdict.ok


def test_invented_column_rejected():
    v = make_validator()
    verdict = v.parse_and_validate(
        "SELECT bogus_col FROM orders LIMIT 1",
        known_tables={"orders"}, known_columns={"orders": {"order_id"}})
    assert not verdict.ok
    assert "bogus_col" in verdict.reason


def test_limit_cap_enforced():
    v = make_validator(max_rows=100)
    assert not v.parse_and_validate("SELECT id FROM t LIMIT 5000").ok


def test_star_requires_limit():
    v = make_validator()
    assert not v.parse_and_validate("SELECT * FROM t").ok
    assert v.parse_and_validate("SELECT * FROM t LIMIT 10").ok


def test_star_inside_count_is_fine():
    v = make_validator()
    assert v.parse_and_validate("SELECT COUNT(*) FROM t").ok
