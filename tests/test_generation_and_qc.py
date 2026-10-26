"""Tests for question/SQL generation, QC dedup and leakage-safe split."""

from dataagent.generation.models import SQL_OUTPUT_SCHEMA, SQLCandidate
from dataagent.generation.questions import (
    TemplateQuestionEngine,
    question_fingerprint,
)
from dataagent.generation.sql_gen import TemplateSQLGenerator
from dataagent.llm.base import extract_json
from dataagent.quality.qc import normalize_sql, sql_fingerprint, split_grouped


def test_template_engine_covers_categories_and_levels(tmp_store):
    engine = TemplateQuestionEngine(tmp_store)
    questions = engine.generate(64)
    categories = {q.category for q in questions}
    difficulties = {q.difficulty for q in questions}
    assert len(categories) >= 5
    assert difficulties == {"beginner", "intermediate", "advanced"}
    # every question records its intended task, tables, columns and pattern
    for q in questions:
        assert q.question
        assert q.task_type or q.category in ("schema_ambiguity", "missing_info", "unsupported")
        assert q.expected_sql_pattern


def test_no_near_duplicates(tmp_store):
    engine = TemplateQuestionEngine(tmp_store)
    questions = engine.generate(200)
    fps = [question_fingerprint(q.question) for q in questions]
    assert len(fps) == len(set(fps)), "near-duplicate question leaked through"


def test_template_sql_parses(tmp_store):
    import sqlglot

    engine = TemplateQuestionEngine(tmp_store)
    generator = TemplateSQLGenerator(tmp_store)
    produced = 0
    for q in engine.generate(200):
        candidate = generator.generate(q)
        if candidate is None:  # intentionally unanswerable categories
            assert q.category in ("schema_ambiguity", "missing_info", "unsupported", "joins")
            continue
        sqlglot.parse_one(candidate.sql, read="dremio")
        produced += 1
    assert produced >= 12


def test_sql_output_schema():
    candidate = {
        "question": "Calculate total completed-order revenue.",
        "required_tables": ['"Analytics"."sales"."orders"'],
        "required_columns": ["amount", "status"],
        "sql": "SELECT SUM(amount) FROM x WHERE status = 'COMPLETED'",
        "task_type": "aggregation",
        "difficulty": "beginner",
    }
    import jsonschema

    jsonschema.validate(candidate, SQL_OUTPUT_SCHEMA)
    candidate["sql"] = 42
    import pytest

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(candidate, SQL_OUTPUT_SCHEMA)


def test_extract_json_handles_fences():
    text = '```json\n{"a": 1}\n```\n some prose'
    assert extract_json(text) == {"a": 1}


def test_sql_fingerprint_normalizes(tmp_store):
    a = "SELECT   order_id FROM orders  WHERE status = 'COMPLETED'"
    b = "select ORDER_ID from ORDERS where STATUS = 'COMPLETED'"
    assert sql_fingerprint(a) == sql_fingerprint(b)


def test_split_groups_templates_together(tmp_store):
    examples = []
    for i in range(8):
        examples.append({
            "example_id": f"ex_{i:03d}",
            "sql_fingerprint": "fp-A" if i < 4 else "fp-B",
            "schema_group": "shop.public.orders",
        })
    splits = split_grouped(examples)
    # every fingerprint must live in exactly one split
    for split_name, members in splits.items():
        fps = {e["sql_fingerprint"] for e in members}
        for other_name, other in splits.items():
            if other_name != split_name:
                assert fps.isdisjoint({e["sql_fingerprint"] for e in other})


def test_normalize_sql_quotes():
    assert normalize_sql("SELECT a FROM t LIMIT 1").startswith("select a from t limit 1")


def test_library_cross_run_dedup(tmp_store, tmp_path, monkeypatch):
    """Two sequential runs: second run must not duplicate the first."""
    import dataagent.pipeline as pipeline_mod
    from dataagent.config import Settings
    from dataagent.quality.library import ExampleLibrary

    settings = Settings(glm_mode="mock", enable_query_execution=False,
                        dremio_user="u", dremio_password="p", dremio_uri="http://mock")
    library = ExampleLibrary(db_path=tmp_path / "lib.sqlite3")
    monkeypatch.setattr("dataagent.pipeline.get_settings", lambda: settings)

    def make_run():
        p = pipeline_mod.DataGenerationPipeline.__new__(pipeline_mod.DataGenerationPipeline)
        p._settings = settings
        p._store = tmp_store
        p._model = pipeline_mod.build_model_client(settings)
        p._mock = True
        p._audit = pipeline_mod.AuditLog(tmp_path / "audit.jsonl")
        p._executor = pipeline_mod.ExecutionService(settings=settings)

        class StubExec:
            def execute(self, sql, example_id=""):
                from dataagent.validation.executor import ExecutionOutcome
                return ExecutionOutcome(executed=True, status="completed", rows=[], row_count=0)
            def enforce_limits(self, sql):
                return sql
        p._executor = StubExec()
        p._correction = pipeline_mod.CorrectionEngine(p._model, settings, max_retries=0)
        p._usage = pipeline_mod.UsageTracker()
        p._evidence = None
        p._library = library
        p._semantic = pipeline_mod.SemanticDedup(persist_dir=tmp_path / "sem_idx")
        p._export_dir = tmp_path
        p._reports_dir = tmp_path
        return p

    p1 = make_run()
    out1 = p1.run(target=20)
    assert out1["added_this_run"] > 0
    assert out1["cumulative_total"] == out1["added_this_run"]

    p2 = make_run()
    out2 = p2.run(target=20)
    # the template bank is small, so run 2 mostly hits cross-run duplicates;
    # cumulative total must never decrease and no cross-run dup may enter
    assert out2["cumulative_total"] >= out1["cumulative_total"]
    from dataagent.quality.library import ExampleLibrary as L2
    lib = L2(db_path=tmp_path / "lib.sqlite3")
    fps = [e["sql_fingerprint"] for e in lib.all_examples()]
    assert len(fps) == len(set(fps)), "duplicate SQL fingerprints in library"
    qfps = [e["question_fingerprint"] for e in lib.all_examples()]
    assert len(qfps) == len(set(qfps)), "duplicate questions in library"
