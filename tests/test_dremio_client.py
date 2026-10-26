"""Tests for the Dremio REST client using a mocked transport (no network)."""

from __future__ import annotations

import httpx
import pytest

from dataagent.dremio.client import (
    AuthenticationError,
    DremioClient,
    PermissionError,
    ResourceUnavailableError,
)


class MockTransport(httpx.BaseTransport):
    """Answers the endpoints used by DremioClient with canned responses."""

    def __init__(self, login_ok: bool = True):
        self.login_ok = login_ok
        self.jobs = {"job-1": {"jobState": "COMPLETED"}}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apiv2/login":
            if self.login_ok:
                return httpx.Response(200, json={"token": "tok123"})
            return httpx.Response(403, json={"errorMessage": "denied"})
        if path == "/api/v3/catalog":
            return httpx.Response(200, json={"data": [
                {"id": "src-1", "path": ["shop"], "type": "CONTAINER",
                 "containerType": "SOURCE"},
            ]})
        if path.startswith("/api/v3/catalog/src-1"):
            return httpx.Response(200, json={
                "id": "src-1", "path": ["shop"], "type": "CONTAINER",
                "containerType": "SOURCE",
                "children": [{"id": "ds-1", "path": ["shop", "orders"],
                              "type": "DATASET", "datasetType": "DIRECT"}]})
        if path.startswith("/api/v3/catalog/by-path/shop/orders"):
            return httpx.Response(200, json={
                "id": "ds-1", "path": ["shop", "orders"], "type": "PHYSICAL_DATASET",
                "fields": [{"name": "order_id", "type": {"name": "BIGINT"}},
                           {"name": "status", "type": {"name": "VARCHAR"}}]})
        if path == "/api/v3/sql":
            return httpx.Response(200, json={"id": "job-1"})
        if path.startswith("/api/v3/job/job-1"):
            if request.url.path.endswith("/results"):
                return httpx.Response(200, json={
                    "schema": [{"name": "n"}], "rows": [{"n": 42}]})
            return httpx.Response(200, json={"jobState": "COMPLETED", "rowCount": 1})
        if path == "/api/v3/forbidden":
            return httpx.Response(403, json={"errorMessage": "nope"})
        if path == "/api/v3/missing":
            return httpx.Response(404, json={"errorMessage": "gone"})
        return httpx.Response(400, json={"errorMessage": "bad request"})


def make_client(login_ok: bool = True) -> DremioClient:
    from dataagent.config import Settings

    settings = Settings(dremio_uri="http://mock", dremio_user="u", dremio_password="p")
    client = DremioClient(settings)
    client._http = httpx.Client(base_url="http://mock", transport=MockTransport(login_ok))
    return client


def test_login_and_catalog():
    client = make_client()
    entries = client.list_catalog()
    assert entries[0].path == ["shop"]
    client.close()


def test_login_failure_raises():
    client = make_client(login_ok=False)
    with pytest.raises(AuthenticationError):
        client.list_catalog()
    client.close()


def test_dataset_by_path():
    client = make_client()
    ds = client.get_dataset_by_path(["shop", "orders"])
    assert [c.name for c in ds.columns] == ["order_id", "status"]
    client.close()


def test_execute_sql_collects_rows():
    client = make_client()
    result = client.execute_sql("SELECT COUNT(*) AS n FROM orders")
    assert result.state == "COMPLETED"
    assert result.rows == [{"n": 42}]
    client.close()


def test_permission_error():
    client = make_client()
    with pytest.raises(PermissionError):
        client._request("GET", "/api/v3/forbidden")
    client.close()


def test_resource_unavailable():
    client = make_client()
    with pytest.raises(ResourceUnavailableError):
        client._request("GET", "/api/v3/missing")
    client.close()
