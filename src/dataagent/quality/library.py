"""Cumulative example library: persistent, deduplicated across runs.

Verified examples are stored in SQLite keyed by:
  - sql_fingerprint   (normalized SQL — exact/normalized duplicates)
  - question_fingerprint (lexical near-duplicate questions)

Each generation run appends only NEW examples; train/validation/test JSONL
files are rebuilt from the library so they always contain the full verified
set (grouped split prevents template leakage).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from dataagent.config import DATA_DIR
from dataagent.quality.qc import split_grouped, sql_fingerprint
from dataagent.generation.questions import question_fingerprint

_SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    example_id TEXT UNIQUE,
    question TEXT NOT NULL,
    assistant_sql TEXT NOT NULL,
    user_content TEXT NOT NULL,
    category TEXT,
    difficulty TEXT,
    task_type TEXT,
    sql_fingerprint TEXT UNIQUE,
    question_fingerprint TEXT UNIQUE,
    schema_group TEXT,
    validation_status TEXT,
    execution_status TEXT,
    semantic_status TEXT,
    reference_comparison TEXT,
    generation_model TEXT,
    dataset_version TEXT,
    created_at TEXT NOT NULL
);
"""


class ExampleLibrary:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or DATA_DIR / "example_library.sqlite3"
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    def reset(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM examples")

    # -- fingerprints ----------------------------------------------------------

    def seen_sql_fingerprints(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT sql_fingerprint FROM examples").fetchall()
        return {r["sql_fingerprint"] for r in rows}

    def seen_question_fingerprints(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT question_fingerprint FROM examples").fetchall()
        return {r["question_fingerprint"] for r in rows}

    def fingerprint_exists(self, sql_fp: str, q_fp: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM examples WHERE sql_fingerprint = ? OR question_fingerprint = ?",
                (sql_fp, q_fp),
            ).fetchone()
        return row is not None

    # -- add / list -------------------------------------------------------------

    def add_examples(self, examples: list[dict]) -> int:
        """Insert verified examples; skip ones already stored. Returns added count."""
        added = 0
        with self._lock, self._conn:
            for ex in examples:
                s_fp = ex.get("sql_fingerprint") or sql_fingerprint(ex["assistant_sql"])
                q_fp = ex.get("question_fingerprint") or question_fingerprint(ex["question"])
                if s_fp is None or q_fp is None:
                    continue
                exists = self._conn.execute(
                    "SELECT 1 FROM examples WHERE sql_fingerprint = ? OR question_fingerprint = ?",
                    (s_fp, q_fp)).fetchone()
                if exists:
                    continue
                example_id = ex.get("example_id") or f"lib_{uuid.uuid4().hex[:12]}"
                self._conn.execute(
                    "INSERT OR IGNORE INTO examples (example_id, question, assistant_sql,"
                    " user_content, category, difficulty, task_type, sql_fingerprint,"
                    " question_fingerprint, schema_group, validation_status,"
                    " execution_status, semantic_status, reference_comparison,"
                    " generation_model, dataset_version, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        example_id,
                        ex["question"],
                        ex["assistant_sql"],
                        ex.get("user_content", ""),
                        ex.get("category"),
                        ex.get("difficulty"),
                        ex.get("task_type"),
                        s_fp,
                        q_fp,
                        ex.get("schema_group"),
                        ex.get("validation_status"),
                        ex.get("execution_status"),
                        ex.get("semantic_status"),
                        ex.get("reference_comparison"),
                        ex.get("generation_model"),
                        ex.get("dataset_version"),
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    ),
                )
                added += 1
        return added

    def all_examples(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM examples ORDER BY category, difficulty, id").fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS n FROM examples").fetchone()["n"]
            by_cat = self._conn.execute(
                "SELECT category, COUNT(*) AS n FROM examples GROUP BY category").fetchall()
        return {"total": total,
                "by_category": {r["category"]: r["n"] for r in by_cat}}

    # -- re-split ---------------------------------------------------------------

    def rebuild_splits(self, out_dir: Path, version: str) -> dict[str, int]:
        """Rebuild train/validation/test SFT JSONL from the full library."""
        from dataagent.export.jsonl import JSONLExporter

        examples = []
        for row in self.all_examples():
            examples.append({
                "example_id": row["example_id"],
                "question": row["question"],
                "assistant_sql": row["assistant_sql"],
                "user_content": row["user_content"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "task_type": row["task_type"],
                "sql_fingerprint": row["sql_fingerprint"],
                "schema_group": row["schema_group"],
                "validation_status": row["validation_status"],
                "execution_status": row["execution_status"],
                "semantic_status": row["semantic_status"],
                "reference_comparison": row["reference_comparison"],
                "generation_model": row["generation_model"],
                "review_status": "verified",
            })
        splits = split_grouped(examples)
        exporter = JSONLExporter(out_dir=out_dir, version=version)
        counts = {}
        for split in ("train", "validation", "test"):
            exporter.write_split(split, splits[split])
            counts[split] = len(splits[split])
        return counts
