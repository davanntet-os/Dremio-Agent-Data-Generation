"""Typed models for Dremio catalog objects and query jobs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CatalogChild(BaseModel):
    id: str
    path: list[str]
    type: str  # CONTAINER | FILE | DATASET | ...
    containerType: str | None = None
    datasetType: str | None = None  # DIRECT | PROMOTED | VIRTUAL


class CatalogEntry(BaseModel):
    id: str
    path: list[str]
    type: str
    containerType: str | None = None
    datasetType: str | None = None
    tag: str | None = None
    children: list[CatalogChild] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ColumnInfo(BaseModel):
    name: str
    type: str
    description: str | None = None


class DatasetInfo(BaseModel):
    """Fully resolved dataset (table or view) with columns."""

    id: str
    path: list[str]
    type: str  # PHYSICAL_DATASET | VIRTUAL_DATASET | FILE ...
    datasetType: str | None = None
    tag: str | None = None
    columns: list[ColumnInfo] = Field(default_factory=list)
    description: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path[-1]

    @property
    def qualified_name(self) -> str:
        """SQL-quoted fully qualified name, e.g. "ecommerce"."orders"."""
        return ".".join(f'"{part}"' for part in self.path)


class JobResult(BaseModel):
    job_id: str
    state: str  # COMPLETED | FAILED | CANCELED | RUNNING ...
    error_message: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    columns: list[str] = Field(default_factory=list)
