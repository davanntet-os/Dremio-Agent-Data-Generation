"""End-to-end pipeline test in mock mode with a stub executor (no network)."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def mock_settings(monkeypatch, tmp_path):
    from dataagent.config import Settings

    settings = Settings(
        glm_mode="mock",
        enable_query_execution=True,
        query_max_rows=100,
        generation_batch_size=16,
        dremio_user="u", dremio_password="p",
        dremio_uri="http://mock",
    )
    monkeypatch.setattr("dataagent.pipeline.get_settings", lambda: settings)
    monkeypatch.setattr("dataagent.export.jsonl.get_settings", lambda: settings)
    monkeypatch.setattr("dataagent.export.jsonl.OUTPUT_DIR", tmp_path)
    return settings


def test_pipeline_mock_end_to_end(tmp_store, mock_settings, monkeypatch, tmp_path):
    import dataagent.pipeline as pipeline_mod

    monkeypatch.setattr("dataagent.pipeline.get_settings", lambda: mock_settings)

    p = pipeline_mod.DataGenerationPipeline.__new__(pipeline_mod.DataGenerationPipeline)
    p._settings = mock_settings
    p._store = tmp_store
    p._model = pipeline_mod.build_model_client(mock_settings)
    p._mock = True
    p._audit = pipeline_mod.AuditLog(tmp_path / "audit.jsonl")
    p._reports_dir = tmp_path

    class StubExecutor:
        def __init__(self):
            self.executed = []

        def execute(self, sql, example_id=""):
            self.executed.append(sql)
            from dataagent.validation.executor import ExecutionOutcome
            return ExecutionOutcome(executed=True, status="completed", rows=[], row_count=0)

        def enforce_limits(self, sql):
            return sql

    stub = StubExecutor()
    p._executor = stub
    p._correction = pipeline_mod.CorrectionEngine(p._model, mock_settings, max_retries=0)
    p._usage = pipeline_mod.UsageTracker()
    p._evidence = None
    p._library = pipeline_mod.ExampleLibrary(db_path=tmp_path / "lib.sqlite3")
    p._semantic = pipeline_mod.SemanticDedup(persist_dir=tmp_path / "sem_idx")
    p._export_dir = tmp_path

    target = 40
    datasets = tmp_store.list_datasets()
    questions, _, failures = p._questions(target, datasets)
    assert len(questions) >= 10

    candidates = p._sql_and_validate(questions, datasets)
    assert candidates

    accepted, rejected, clarifications = p._verify_qc(candidates)
    assert accepted
    split_counts, outputs = p._export(accepted, clarifications)
    assert split_counts["train"] > split_counts["validation"]
    assert sum(split_counts[s] for s in ("train", "validation", "test")) == \
        len([e for e in accepted if not e.get("needs_review")])

    train = [json.loads(line) for line in open(tmp_path / "train.jsonl")]
    for example in train:
        assert example["messages"][0]["role"] == "system"
        assert example["messages"][-1]["role"] == "assistant"
        content = example["messages"][-1]["content"]
        assert "example_id" not in content, "QC metadata leaked into SFT target"
        assert content.strip().upper().startswith("SELECT") or content.strip().upper().startswith("WITH")
