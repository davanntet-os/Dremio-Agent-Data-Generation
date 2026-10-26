"""Dremio REST API v3 client.

Covers authentication (PAT or user/password login token), catalog traversal,
dataset details, SQL submission, job polling and paginated result retrieval.

The client performs no SQL-safety logic itself; execution always flows through
dataagent.validation.executor, which enforces read-only rules, timeouts and
row limits.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import quote

import httpx

from dataagent.config import Settings, get_settings
from dataagent.dremio.models import CatalogChild, CatalogEntry, DatasetInfo, JobResult
from dataagent.logging_utils import get_logger

log = get_logger(__name__)

POLL_INTERVAL_SECONDS = 1.0


class DremioError(Exception):
    pass


class AuthenticationError(DremioError):
    pass


class PermissionError(DremioError):  # noqa: A001 - deliberate domain name
    pass


class ResourceUnavailableError(DremioError):
    pass


class DremioClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._http = httpx.Client(
            base_url=self._settings.dremio_uri.rstrip("/"),
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )
        self._token: str | None = self._settings.dremio_pat or None
        self._token_lock = threading.Lock()
        self._metadata = {"User-Agent": "dataagent/0.1.0"}

    def __enter__(self) -> "DremioClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- authentication ----------------------------------------------------

    def _ensure_token(self) -> str:
        with self._token_lock:
            if self._token:
                return self._token
            payload = {"userName": self._settings.dremio_user,
                       "password": self._settings.dremio_password}
            try:
                resp = self._http.post("/apiv2/login", json=payload)
            except httpx.HTTPError as exc:
                raise ResourceUnavailableError(f"Dremio unreachable: {exc}") from exc
            if resp.status_code in (401, 403):
                detail = resp.json().get("errorMessage", resp.text)
                raise AuthenticationError(f"Dremio login rejected: {detail}")
            if resp.status_code != 200:
                raise DremioError(f"Dremio login failed ({resp.status_code}): {resp.text}")
            self._token = resp.json()["token"]
            return self._token

    def _headers(self) -> dict[str, str]:
        return self._metadata | {"Authorization": f"_dremio{self._ensure_token()}"}

    def close(self) -> None:
        self._http.close()

    # -- low-level request -------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._http.request(method, path, headers=self._headers(), **kwargs)
        if resp.status_code in (401, 403):
            raise PermissionError(f"Dremio returned {resp.status_code}: {resp.text[:300]}")
        if resp.status_code == 404:
            raise ResourceUnavailableError(f"Not found: {path}")
        if resp.status_code >= 400:
            raise DremioError(f"Dremio error {resp.status_code} on {path}: {resp.text[:500]}")
        if not resp.content:
            return None
        return resp.json()

    # -- catalog ------------------------------------------------------------

    def list_catalog(self) -> list[CatalogEntry]:
        data = self._request("GET", "/api/v3/catalog")
        return [CatalogEntry(id=o["id"], path=o["path"], type=o["type"],
                             containerType=o.get("containerType"), raw=o)
                for o in data.get("data", [])]

    def get_catalog_entity(self, entity_id: str) -> CatalogEntry:
        obj = self._request("GET", f"/api/v3/catalog/{quote(entity_id, safe='')}")
        # HOME entities come back without a `path` or `type` array.
        path = obj.get("path") or [obj.get("name", entity_id)]
        return CatalogEntry(
            id=obj["id"],
            path=path,
            type=obj.get("type", "CONTAINER"),
            containerType=obj.get("containerType"),
            datasetType=obj.get("datasetType"),
            tag=obj.get("tag"),
            children=[CatalogChild(**c) for c in obj.get("children", [])],
            raw=obj,
        )

    def get_dataset_by_path(self, path: list[str]) -> DatasetInfo:
        quoted = "/".join(quote(part, safe="") for part in path)
        obj = self._request("GET", f"/api/v3/catalog/by-path/{quoted}")
        return self._to_dataset_info(obj)

    @staticmethod
    def _to_dataset_info(obj: dict[str, Any]) -> DatasetInfo:
        fields = obj.get("fields") or []
        columns = [
            {"name": f.get("name", ""), "type": f.get("type", {}).get("name", ""),
             "description": f.get("description")}
            if isinstance(f.get("type"), dict)
            else {"name": f.get("name", ""), "type": str(f.get("type") or ""), "description": f.get("description")}
            for f in fields
        ]
        return DatasetInfo(
            id=obj["id"],
            path=obj["path"],
            type=obj.get("type", ""),
            datasetType=obj.get("datasetType"),
            tag=obj.get("tag"),
            description=obj.get("description"),
            raw=obj,
            columns=columns,  # type: ignore[arg-type]
        )

    # -- sources (bootstrap only) -------------------------------------------

    def create_source(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v3/source", json=body)

    def list_sources(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/v3/source")
        return data.get("data", [])

    def start_metadata_refresh(self, source_name: str) -> None:
        """Optimistically fetch metadata for every dataset in a source."""
        self._request("POST", f"/api/v3/catalog/source/{source_name}/refresh", json={})

    # -- SQL execution --------------------------------------------------------

    def submit_sql(self, sql: str, context: list[str] | None = None) -> str:
        body: dict[str, Any] = {"sql": sql}
        if context:
            body["context"] = context
        resp = self._request("POST", "/api/v3/sql", json=body)
        return resp["id"]

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v3/job/{job_id}")

    def execute_sql(self, sql: str, timeout_seconds: float = 60.0) -> JobResult:
        job_id = self.submit_sql(sql)
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = self.get_job_status(job_id)
            state = status.get("jobState", status.get("state", ""))
            if state == "COMPLETED":
                return self._collect_results(job_id, status)
            if state in ("FAILED", "CANCELED", "Canceled",
                         "Cancellation requested"):
                return JobResult(
                    job_id=job_id,
                    state=state,
                    error_message=status.get("errorMessage") or None,
                )
            if time.monotonic() > deadline:
                self.cancel_job(job_id)
                return JobResult(job_id=job_id, state="TIMED_OUT", error_message="Execution timeout")
            time.sleep(POLL_INTERVAL_SECONDS)
        # unreachable

    def cancel_job(self, job_id: str) -> None:
        try:
            self._request("POST", f"/api/v3/job/{job_id}/cancel", json={})
        except DremioError as exc:
            log.warning("cancel_job(%s) failed: %s", job_id, exc)

    def _collect_results(self, job_id: str, status: dict[str, Any]) -> JobResult:
        offset, limit = 0, 500
        rows: list[dict[str, Any]] = []
        columns: list[str] = []
        while True:
            data = self._request(
                "GET",
                f"/api/v3/job/{job_id}/results",
                params={"offset": offset, "limit": limit},
            )
            schema = data.get("schema") or []
            if not columns and schema:
                columns = [c.get("name", "") for c in schema]
            batch = data.get("rows", [])
            rows.extend(batch)
            offset += len(batch)
            if len(batch) < limit or offset >= 10_000:
                break
        return JobResult(
            job_id=job_id,
            state=status.get("state", "COMPLETED"),
            rows=rows,
            row_count=status.get("rowCount", len(rows)),
            columns=columns,
        )
