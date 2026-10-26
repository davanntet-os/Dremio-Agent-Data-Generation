"""Synthetic test fixtures: schemas and a mock Dremio transport."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("GLM_MODE", "mock")
os.environ.setdefault("ENABLE_QUERY_EXECUTION", "false")
os.environ.setdefault("DREMIO_PASSWORD", "test")
os.environ.setdefault("DREMIO_ADMIN_PASSWORD", "test")

SYNTHETIC_DATASETS = [
    {
        "fingerprint": "shop.public.orders",
        "path": ["shop", "public", "orders"],
        "name": "orders",
        "dataset_type": "PHYSICAL_DATASET",
        "description": None,
        "dremio_tag": "tag1",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "raw": {},
        "columns": [
            {"name": "order_id", "type": "BIGINT", "description": None},
            {"name": "customer_id", "type": "BIGINT", "description": None},
            {"name": "order_date", "type": "TIMESTAMP", "description": None},
            {"name": "status", "type": "VARCHAR", "description": None},
            {"name": "total_amount", "type": "DECIMAL", "description": None},
        ],
    },
    {
        "fingerprint": "shop.public.customers",
        "path": ["shop", "public", "customers"],
        "name": "customers",
        "dataset_type": "PHYSICAL_DATASET",
        "description": None,
        "dremio_tag": "tag1",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "raw": {},
        "columns": [
            {"name": "customer_id", "type": "BIGINT", "description": None},
            {"name": "email", "type": "VARCHAR", "description": None},
            {"name": "region", "type": "VARCHAR", "description": None},
        ],
    },
]

SYNTHETIC_RELATIONSHIP = {
    "left": "shop.public.orders",
    "left_columns": ["customer_id"],
    "right": "shop.public.customers",
    "right_columns": ["customer_id"],
    "source": "view_sql",
    "evidence": "test fixture",
}


@pytest.fixture
def tmp_store(tmp_path: Path):
    from dataagent.schema.store import SchemaStore

    store = SchemaStore(db_path=tmp_path / "cache.sqlite3")
    for ds in SYNTHETIC_DATASETS:
        from dataagent.dremio.models import DatasetInfo

        info = DatasetInfo(
            id="id-" + ds["fingerprint"],
            path=ds["path"],
            type=ds["dataset_type"],
            datasetType=ds["dataset_type"],
            tag=ds["dremio_tag"],
            raw=ds["raw"],
        )
        from dataagent.dremio.models import ColumnInfo

        info.columns = [ColumnInfo(**c) for c in ds["columns"]]
        store.upsert_dataset(info)
    store.add_relationship(
        SYNTHETIC_RELATIONSHIP["left"],
        SYNTHETIC_RELATIONSHIP["left_columns"],
        SYNTHETIC_RELATIONSHIP["right"],
        SYNTHETIC_RELATIONSHIP["right_columns"],
        SYNTHETIC_RELATIONSHIP["source"],
        SYNTHETIC_RELATIONSHIP["evidence"],
    )
    yield store
    store.close()
