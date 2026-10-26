"""Bootstrap the local Dremio dev environment.

Steps:
  1. Wait until Dremio answers.
  2. Create the first admin user (idempotent; ignored if it already exists).
  3. Create the Postgres source `ecommerce` if absent.
  4. Wait until the seeded tables are queryable through Dremio.
  5. Create a dedicated non-admin runtime user for the pipeline.

All credentials come from environment variables (.env). Never printed.
"""

from __future__ import annotations

import os
import sys
import time

import httpx

from dataagent.config import get_settings
from dataagent.logging_utils import get_logger

log = get_logger("bootstrap")

CREATE_FIRST_USER = "/apiv2/bootstrap/firstuser"
LOGIN = "/apiv2/login"


def wait_for_dremio(http: httpx.Client, minutes: float = 8.0) -> None:
    deadline = time.monotonic() + minutes * 60
    while time.monotonic() < deadline:
        try:
            resp = http.get("/api/v3/catalog")
            # 401/403 means the auth layer is up: the API itself is available.
            if resp.status_code in (200, 401, 403):
                return
        except httpx.HTTPError:
            pass
        log.info("Waiting for Dremio to become available...")
        time.sleep(5)
    sys.exit("Dremio did not become available in time")


def create_first_user(http: httpx.Client, user: str, password: str) -> None:
    body = {
        "userName": user,
        "firstName": "Data",
        "lastName": "Agent",
        "email": "dataagent@example.com",
        "createdAt": int(time.time() * 1000),
        "password": password,
    }
    resp = http.put(
        CREATE_FIRST_USER,
        json=body,
        headers={"Authorization": "_dremionull"},
    )
    if resp.status_code in (200, 201):
        log.info("Created first Dremio admin user %s", user)
    else:
        log.info("First-user creation returned %s (probably exists): ok", resp.status_code)


def login(http: httpx.Client, user: str, password: str) -> str:
    resp = http.post(LOGIN, json={"userName": user, "password": password})
    if resp.status_code != 200:
        sys.exit(f"Login failed for {user}: {resp.status_code} {resp.text[:200]}")
    token = resp.json()["token"]
    return f"_dremio{token}"


def create_source(http: httpx.Client, headers: dict, name: str) -> None:
    existing = http.get("/api/v3/source", headers=headers).json().get("data", [])
    if any(s["name"] == name for s in existing):
        log.info("Source %s already exists", name)
        return
    body = {
        "entityType": "source",
        "name": name,
        "description": "Synthetic e-commerce seed data for dataagent",
        "type": "POSTGRES",
        "config": {
            "hostname": "postgres",
            "port": "5432",
            "databaseName": "ecommerce",
            "username": os.getenv("DREMIO_RO_DB_USER", "dremio_ro"),
            "password": os.getenv("DREMIO_RO_DB_PASSWORD", "ro-password"),
        },
    }
    resp = http.post("/api/v3/source", headers=headers, json=body)
    if resp.status_code not in (200, 201):
        sys.exit(f"Source creation failed: {resp.status_code} {resp.text[:300]}")
    log.info("Created Postgres source %s", name)


def warm_metadata(http: httpx.Client, headers: dict, source: str) -> None:
    """Browse source -> folders -> datasets so Dremio fetches metadata."""
    src = http.get(f"/api/v3/catalog/by-path/{source}", headers=headers).json()
    stack = [(child["id"], child["path"]) for child in src.get("children", [])]
    while stack:
        entity_id, path = stack.pop()
        detail = http.get(f"/api/v3/catalog/{entity_id}", headers=headers).json()
        log.info("Fetched metadata for %s (%d fields)",
                 ".".join(path), len(detail.get("fields", [])))
        stack.extend((c["id"], c["path"]) for c in detail.get("children", []))


def wait_until_queryable(http: httpx.Client, headers: dict, source: str, minutes: float = 6.0) -> None:
    warm_metadata(http, headers, source)
    sql = f'SELECT COUNT(*) AS n FROM "{source}"."public"."orders"'
    deadline = time.monotonic() + minutes * 60
    resp = http.post("/api/v3/sql", headers=headers, json={"sql": sql})
    if resp.status_code != 200:
        sys.exit(f"SQL submit failed: {resp.status_code} {resp.text[:200]}")
    job_id = resp.json()["id"]
    while time.monotonic() < deadline:
        status = http.get(f"/api/v3/job/{job_id}", headers=headers).json()
        state = status.get("jobState")
        if state == "COMPLETED":
            log.info("Source %s is queryable (job %s)", source, job_id)
            return
        if state == "FAILED":
            log.info("Waiting for metadata refresh... (last error: %s)",
                     (status.get("errorMessage") or "")[:120])
        time.sleep(2)
    sys.exit("Datasets are not queryable yet; try re-running this script")


def create_runtime_user(http: httpx.Client, headers: dict, name: str, password: str) -> None:
    resp_get = http.get(f"/apiv2/user/{name}", headers=headers)
    if resp_get.status_code == 200:
        log.info("Runtime user %s exists", name)
        return
    resp = http.post(
        "/api/v3/user", headers=headers,
        json={"name": name, "password": password, "email": f"{name}@example.com"},
    )
    if resp.status_code not in (200, 201):
        sys.exit(f"Runtime user creation failed: {resp.status_code} {resp.text[:300]}")
    log.info("Created runtime user %s (read-only: enforced by Postgres role + parser)", name)


JOIN_VIEWS = {
    "v_customer_orders": (
        'SELECT c.customer_id, c.first_name, c.last_name, o.order_id, o.order_date,'
        ' o.status, o.total_amount'
        ' FROM "ecommerce"."public"."customers" c'
        ' JOIN "ecommerce"."public"."orders" o ON o.customer_id = c.customer_id'
    ),
    "v_order_item_returns": (
        'SELECT oi.order_id, oi.product_id, oi.quantity, oi.line_amount,'
        ' r.return_id, r.return_date, r.refund_amount'
        ' FROM "ecommerce"."public"."order_items" oi'
        ' JOIN "ecommerce"."public"."returns" r ON r.order_item_id = oi.order_item_id'
    ),
    "v_revenue_by_region": (
        'SELECT c.region, SUM(o.total_amount) AS region_revenue'
        ' FROM "ecommerce"."public"."customers" c'
        ' JOIN "ecommerce"."public"."orders" o ON o.customer_id = c.customer_id'
        " WHERE o.status = 'COMPLETED' GROUP BY c.region"
    ),
}


def create_definition_views(http: httpx.Client, headers: dict, space: str = "analytics") -> None:
    """Create Dremio virtual datasets whose SQL documents verified joins.

    Virtual datasets expose their defining SQL through the catalog API, giving
    the discovery service metadata-backed (never name-inferred) relationship
    evidence.
    """
    resp = http.post("/api/v3/catalog", headers=headers,
                     json={"entityType": "space", "name": space})
    if resp.status_code not in (200, 201):
        log.info("Space %s: %s (probably exists)", space, resp.status_code)
    for name, sql in JOIN_VIEWS.items():
        body = {
            "entityType": "dataset",
            "type": "VIRTUAL_DATASET",
            "path": [space, name],
            "sql": sql,
        }
        resp = http.post("/api/v3/catalog", headers=headers, json=body)
        if resp.status_code in (200, 201):
            log.info("Created virtual dataset %s.%s", space, name)
        else:
            log.info("Virtual dataset %s: %s (probably exists)", name, resp.status_code)


def main() -> None:
    settings = get_settings()
    with httpx.Client(base_url=settings.dremio_uri.rstrip("/"), timeout=20.0) as http:
        wait_for_dremio(http)
        create_first_user(http, settings.dremio_admin_user, settings.dremio_admin_password)
        admin_headers = {"Authorization": login(http, settings.dremio_admin_user, settings.dremio_admin_password)}
        create_source(http, admin_headers, "ecommerce")
        wait_until_queryable(http, admin_headers, "ecommerce")
        create_definition_views(http, admin_headers)
        create_runtime_user(http, admin_headers, settings.dremio_user, settings.dremio_password)
    print("bootstrap: OK")


if __name__ == "__main__":
    main()
