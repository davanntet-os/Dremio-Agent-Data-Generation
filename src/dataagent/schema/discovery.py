"""Schema discovery service.

Walks the Dremio catalog (sources, spaces, folders, datasets, views) and
populates the schema metadata cache. Only *documented* sources of evidence
(catalog metadata, wiki/tags descriptions, dataset-level SQL of virtual
datasets) are used for relationships; column-name similarity alone is never
treated as proof of a foreign key. Relationship evidence from virtual-view
SQL (e.g. views whose definitions join tables) is recorded explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataagent.dremio.client import DremioClient, DremioError
from dataagent.dremio.models import CatalogChild, CatalogEntry
from dataagent.logging_utils import get_logger
from dataagent.schema.store import SchemaStore

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class SchemaDiscoveryService:
    def __init__(self, client: DremioClient | None = None) -> None:
        from dataagent.config import get_settings

        self._client = client or DremioClient()
        self._store = SchemaStore()
        self._settings = get_settings()

    @property
    def store(self) -> SchemaStore:
        return self._store

    def discover(self, include_samples: bool = True) -> dict[str, int]:
        """Walk the catalog and refresh the cache.

        Returns counters describing what was updated. Datasets whose Dremio
        `tag` is unchanged are skipped (metadata-cache hit).
        """
        counters = {"datasets": 0, "columns": 0, "skipped": 0, "errors": 0}
        roots = self._client.list_catalog()
        for root in roots:
            # Expand the container to obtain children (list_catalog omits them).
            try:
                entry = self._client.get_catalog_entity(root.id) if not root.children else root
            except DremioError as exc:
                counters["errors"] += 1
                log.warning("Failed to expand %s: %s", root.path, exc)
                continue
            self._walk(entry, counters, depth=0)
        self._derive_virtual_relationships(counters)
        self._store.close()
        return counters

    MAX_DEPTH = 8

    def _walk(self, entry: CatalogEntry, counters: dict[str, int], depth: int = 0) -> None:
        is_dataset = entry.type in ("PHYSICAL_DATASET", "VIRTUAL_DATASET",
                                    "PHYSICAL_DATASET_SOURCE_FILE") or (
            entry.datasetType in ("DIRECT", "PROMOTED", "VIRTUAL")
        )
        if is_dataset:
            try:
                self._update_dataset(entry, counters)
            except DremioError as exc:
                counters["errors"] += 1
                log.warning("Failed to fetch dataset %s: %s", entry.path, exc)
            return
        if depth >= self.MAX_DEPTH:
            return
        for child in entry.children:
            self._walk_child(child, counters, depth + 1)

    def _walk_child(self, child: CatalogChild, counters: dict[str, int], depth: int) -> None:
        is_dataset = child.type == "DATASET" or child.datasetType in ("DIRECT", "PROMOTED", "VIRTUAL")
        if is_dataset:
            try:
                self._update_dataset_by_path(child, counters)
            except DremioError as exc:
                counters["errors"] += 1
                log.warning("Failed to fetch dataset %s: %s", child.path, exc)
            return
        if depth >= self.MAX_DEPTH:
            return
        try:
            inner = self._client.get_catalog_entity(child.id)
        except DremioError as exc:
            counters["errors"] += 1
            log.warning("Failed to expand %s: %s", child.path, exc)
            return
        self._walk(inner, counters, depth)

    def _update_dataset_by_path(self, child: CatalogChild, counters: dict[str, int]) -> None:
        cached_tag = self._store.get_tag(child.path)
        dataset = self._client.get_dataset_by_path(child.path)
        if cached_tag is not None and dataset.tag == cached_tag:
            counters["skipped"] += 1
            return
        self._store_dataset(dataset, counters)

    def _update_dataset(self, entry: CatalogEntry, counters: dict[str, int]) -> None:
        cached_tag = self._store.get_tag(entry.path)
        dataset = self._client.get_dataset_by_path(entry.path)
        if cached_tag is not None and dataset.tag == cached_tag:
            counters["skipped"] += 1
            return
        self._store_dataset(dataset, counters)

    def _store_dataset(self, dataset, counters: dict[str, int]) -> None:
        fp = self._store.upsert_dataset(dataset)
        counters["datasets"] += 1
        counters["columns"] += len(dataset.columns)
        log.info("Cached dataset %s (%d columns)", ".".join(dataset.path), len(dataset.columns))

    def _derive_virtual_relationships(self, counters: dict[str, int]) -> None:
        """Record relationships evidenced by virtual dataset (view) SQL.

        A view's definition is documented evidence of how tables relate, so
        join conditions found there are treated as verified. Column-name
        similarity alone is never recorded.
        """
        import sqlglot
        from sqlglot import exp as E

        for ds in self._store.list_datasets():
            if ds["dataset_type"] not in ("VIRTUAL", "VIRTUAL_DATASET"):
                continue
            sql = (ds.get("raw") or {}).get("sql") or ""
            if not sql or "JOIN" not in sql.upper():
                continue
            try:
                tree = sqlglot.parse_one(sql, read="dremio")
            except Exception:
                continue
            alias_to_table: dict[str, str] = {}
            for tbl in tree.find_all(E.Table):
                alias = (tbl.alias_or_name or tbl.name).lower()
                dotted = ".".join((p.name if isinstance(p, E.Identifier) else str(p)).lower()
                                  for p in tbl.parts) if tbl.parts else tbl.name.lower()
                alias_to_table[alias] = dotted
            for join in tree.find_all(E.Join):
                cond = join.args.get("on")
                if not isinstance(cond, E.EQ):
                    continue
                left, right = cond.this, cond.expression
                if not (isinstance(left, E.Column) and isinstance(right, E.Column)):
                    continue
                lt, rt = left.table.lower(), right.table.lower()
                if lt not in alias_to_table or rt not in alias_to_table:
                    continue
                left_key = alias_to_table[lt] + "." + left.name.lower()
                right_key = alias_to_table[rt] + "." + right.name.lower()
                # Fingerprints match on the TABLE part only.
                fps = self._find_fingerprint(alias_to_table[lt]), self._find_fingerprint(alias_to_table[rt])
                if not all(fps):
                    continue
                self._store.add_relationship(
                    left_fingerprint=fps[0],
                    left_columns=[left.name],
                    right_fingerprint=fps[1],
                    right_columns=[right.name],
                    source="view_sql",
                    evidence=f'view {ds["name"]}: {".".join(ds["path"])}',
                )
                counters["relationships"] = counters.get("relationships", 0) + 1
                log.info("Recorded verified join %s = %s (from view %s)",
                         left_key, right_key, ds["name"])

    def _find_fingerprint(self, dotted_key: str) -> str | None:
        """Match `db.schema.table or schema.table or table` against cache."""
        parts = dotted_key.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:]).lower()
            for ds in self._store.list_datasets():
                if ds["fingerprint"] == suffix or ds["fingerprint"].endswith("." + suffix):
                    return ds["fingerprint"]
        return None
