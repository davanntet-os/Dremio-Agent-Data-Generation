"""SQLite-backed schema metadata cache.

Stores fully qualified dataset paths, table/view names, columns, types,
descriptions, verified relationships and retrieval timestamps. Dremio's
catalog `tag` is used for change detection so refresh can avoid unnecessary
requests.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from dataagent.config import SCHEMA_CACHE_DB
from dataagent.dremio.models import DatasetInfo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    fingerprint TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    dataset_type TEXT,
    description TEXT,
    dremio_tag TEXT,
    raw_json TEXT,
    retrieved_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS columns (
    fingerprint TEXT NOT NULL REFERENCES datasets(fingerprint) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    description TEXT,
    PRIMARY KEY (fingerprint, ordinal)
);
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    left_fingerprint TEXT NOT NULL,
    left_columns TEXT NOT NULL,
    right_fingerprint TEXT NOT NULL,
    right_columns TEXT NOT NULL,
    source TEXT NOT NULL,
    evidence TEXT
);
CREATE TABLE IF NOT EXISTS glossary (
    dataset_fingerprint TEXT PRIMARY KEY REFERENCES datasets(fingerprint) ON DELETE CASCADE,
    definition TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_columns_name ON columns(name);
"""


def _fingerprint(path: list[str]) -> str:
    return ".".join(part.lower() for part in path)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class SchemaStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or SCHEMA_CACHE_DB
        # Shared connection used from pipeline worker threads; serialized by lock.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    # -- datasets ------------------------------------------------------------

    def upsert_dataset(self, dataset: DatasetInfo) -> str:
        fp = _fingerprint(dataset.path)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO datasets (fingerprint, path, name, dataset_type, description,"
                " dremio_tag, raw_json, retrieved_at)"
                " VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(fingerprint) DO UPDATE SET name=excluded.name,"
                " dataset_type=excluded.dataset_type, description=excluded.description,"
                " dremio_tag=excluded.dremio_tag, raw_json=excluded.raw_json,"
                " retrieved_at=excluded.retrieved_at",
                (
                    fp,
                    json.dumps(dataset.path),
                    dataset.name,
                    dataset.datasetType or dataset.type,
                    dataset.description,
                    dataset.tag,
                    json.dumps(dataset.raw, default=str),
                    now_utc(),
                ),
            )
            self._conn.execute("DELETE FROM columns WHERE fingerprint = ?", (fp,))
            for ordinal, col in enumerate(dataset.columns):
                self._conn.execute(
                    "INSERT INTO columns (fingerprint, ordinal, name, type, description)"
                    " VALUES (?,?,?,?,?)",
                    (fp, ordinal, col.name, col.type, col.description),
                )
        return fp

    def get_tag(self, path: list[str]) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT dremio_tag FROM datasets WHERE fingerprint = ?",
                (_fingerprint(path),),
            ).fetchone()
        return row["dremio_tag"] if row else None

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM datasets ORDER BY path").fetchall()
            out = []
            for row in rows:
                cols = self._conn.execute(
                    "SELECT name, type, description FROM columns"
                    " WHERE fingerprint = ? ORDER BY ordinal",
                    (row["fingerprint"],),
                ).fetchall()
                out.append(
                    {
                        "fingerprint": row["fingerprint"],
                        "path": json.loads(row["path"]),
                        "name": row["name"],
                        "dataset_type": row["dataset_type"],
                        "description": row["description"],
                        "dremio_tag": row["dremio_tag"],
                        "retrieved_at": row["retrieved_at"],
                        "columns": [dict(c) for c in cols],
                        "raw": json.loads(row["raw_json"]) if row["raw_json"] else {},
                    }
                )
        return out

    def find_datasets_by_column(self, column_name: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT d.fingerprint FROM datasets d"
                " JOIN columns c ON c.fingerprint = d.fingerprint"
                " WHERE lower(c.name) = lower(?)",
                (column_name,),
            ).fetchall()
        fps = [r["fingerprint"] for r in rows]
        return [d for d in self.list_datasets() if d["fingerprint"] in fps]

    def get_dataset(self, fingerprint: str) -> dict[str, Any] | None:
        for d in self.list_datasets():
            if d["fingerprint"] == fingerprint:
                return d
        return None

    # -- relationships (metadata / documented only) ----------------------------

    def add_relationship(
        self,
        left_fingerprint: str,
        left_columns: list[str],
        right_fingerprint: str,
        right_columns: list[str],
        source: str,
        evidence: str = "",
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO relationships"
                " (left_fingerprint, left_columns, right_fingerprint, right_columns,"
                "  source, evidence) VALUES (?,?,?,?,?,?)",
                (
                    left_fingerprint,
                    json.dumps(left_columns),
                    right_fingerprint,
                    json.dumps(right_columns),
                    source,
                    evidence,
                ),
            )

    def list_relationships(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM relationships").fetchall()
        return [
            {
                "left": r["left_fingerprint"],
                "left_columns": json.loads(r["left_columns"]),
                "right": r["right_fingerprint"],
                "right_columns": json.loads(r["right_columns"]),
                "source": r["source"],
                "evidence": r["evidence"],
            }
            for r in rows
        ]

    # -- glossary --------------------------------------------------------------

    def upsert_glossary(self, fingerprint: str, definition: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO glossary (dataset_fingerprint, definition) VALUES (?,?)"
                " ON CONFLICT(dataset_fingerprint) DO UPDATE SET definition=excluded.definition",
                (fingerprint, definition),
            )

    def list_glossary(self) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM glossary").fetchall()
        return [{"fingerprint": r["dataset_fingerprint"], "definition": r["definition"]} for r in rows]
