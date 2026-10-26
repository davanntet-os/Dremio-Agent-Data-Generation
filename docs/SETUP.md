# Setup Guide

## 1. Local dev stack (Dremio + Postgres + dremio-mcp)

```bash
cp .env.example .env            # then edit credentials
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

docker compose -f docker/docker-compose.yml --profile mcp up -d
python scripts/bootstrap_dremio.py
```

The bootstrap script is idempotent and:

1. waits for Dremio to become reachable,
2. creates the first admin user (`DREMIO_ADMIN_USER/PASSWORD`) via
   `PUT /apiv2/bootstrap/firstuser`,
3. registers the Postgres source `ecommerce` (read-only role `dremio_ro`),
4. warms dataset metadata and waits until `SELECT COUNT(*)` succeeds,
5. creates Dremio *virtual datasets* (`analytics.v_*`) whose SQL documents
   verified join relationships — this is the metadata-backed relationship
   evidence used by schema discovery (column-name matching is never used),
6. creates the non-admin runtime user (`DREMIO_USER`).

## 2. Dremio connection details

| Component | Value |
|---|---|
| REST API v3 | `http://localhost:9047/api/v3` |
| UI | `http://localhost:9047` |
| Runtime auth | login token (`POST /apiv2/login`) or PAT (`DREMIO_PAT`) |
| Read-only enforcement | Postgres `dremio_ro` role (SELECT only) + parser allowlist |

### Connecting an existing Dremio (Software)

Set in `.env`:

```
DREMIO_URI=https://your-dremio-host:9047
DREMIO_PAT=<personal access token>   # recommended
```

Then run `dataagent discover`. All read-only restrictions still apply; the
execution service additionally enforces statement-type allowlists, timeouts
and row limits.

## 3. LLM provider configuration

### OpenCode Go (recommended if subscribed)

Verified facts (see https://opencode.ai/console/go and the Inference API docs):

| Setting | Value |
|---|---|
| `GLM_MODE` | `opencode` |
| Base URL | `https://opencode.ai/inference/go/openai/v1/chat/completions` |
| GLM model ids | `glm-5.3-flash` (cheap) / `glm-5.3` / `glm-5.2` / `glm-5.1` |
| Auth | `Authorization: Bearer <console API key>` (personal `oc_sk_...` or service-account key) |
| Routing header | `x-opencode-session: <stable uuid>` (sent automatically by the client) |
| Model list | `https://opencode.ai/inference/go/v1/models` |

The API key resolves in this order: `OPENCODE_API_KEY` env var → opencode's
own auth store (`~/.local/share/opencode/auth.json`, provider `opencode-go`,
populated by `/connect`). Nothing is logged or copied into the repo.

Verify before the first run:

```bash
dataagent llm-check
```

### Z.AI / Zhipu direct (alternative)

| Setting | Value |
|---|---|
| `GLM_MODE` | `zai` |
| `GLM_BASE_URL` | `https://api.z.ai/api/paas/v4` (intl) or `https://open.bigmodel.cn/api/paas/v4` (CN) |
| `GLM_MODEL` | `glm-5.3` (flagship) or `glm-5.3-flash` (lower cost) |
| `GLM_REASONING_EFFORT` | `low` / `high` / `max` (reasoning is always on for glm-5.3) |
| `GLM_API_KEY` | Bearer key from https://z.ai/manage-apikey/apikey-list |

### Offline (mock)

`GLM_MODE=mock` uses a deterministic template engine — no network, no key.
Used by tests and to smoke-test the pipeline.

Provider implementations live in `src/dataagent/llm/` and are chosen only in
`build_model_client` so the provider can be swapped via configuration.

## 4. dremio-mcp (optional interactive server)

```bash
git clone https://github.com/dremio/dremio-mcp docker/dremio-mcp
# compose builds it under the `mcp` profile; requires a PAT in DREMIO_MCP_PAT
```

## 5. Pipeline

```bash
dataagent discover                 # cache schemas
dataagent query 'SELECT ...'       # safe read-only smoke test
dataagent generate --target 500    # generate -> validate -> verify -> export
```

Outputs land in `data/output/` (`train|validation|test` SFT JSONL +
`.manifest.jsonl` QC files) and reports in `reports/`.
