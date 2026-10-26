# Dremio Synthetic Training Data Generator ("dataagent")

An AI-powered data-generation agent that uses a GLM model via the
OpenAI-compatible API of your choice (default: **OpenCode Go** subscription;
Z.AI/Zhipu direct also supported) to discover real Dremio database schemas,
generate diverse natural-language analytical questions, generate corresponding
Dremio SQL, validate and execute the queries safely against a test
environment, and export verified training examples in a `messages`-style JSONL
format for fine-tuning Qwen3.5-0.8B.

**Quality over quantity**: unverified examples are never promoted into the
training set.

## Architecture

```
src/dataagent/
  config.py          settings (env-driven, pydantic-settings)
  llm/               model clients (GLM via OpenAI-compatible protocol)
  dremio/            Dremio REST API v3 client
  schema/            discovery service + SQLite metadata cache
  generation/        question engine + SQL engine + prompts
  validation/        7-stage SQL validation, execution, semantic verification
  errors/            Dremio error classification + correction loop
  quality/           dedup, SQL fingerprints, leakage-safe split, reports
  export/            JSONL SFT exporter + separate QC manifest
  eval/              acceptance / rejection reporting
  pipeline.py        orchestrator
  cli.py             argparse CLI
```

Model-specific code is isolated in `dataagent/llm/`; the Dremio integration
never imports model code, so the provider can be swapped via configuration.

## Quickstart (bundled dev stack)

```bash
cp .env.example .env            # fill in GLM_API_KEY etc.
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# 1. Start local Dremio OSS + Postgres seed source (+ dremio-mcp)
docker compose -f docker/docker-compose.yml --profile mcp up -d

# 2. Bootstrap: create source, read-only user, wait for readiness
python scripts/bootstrap_dremio.py

# 3. Discover schemas into the metadata cache
dataagent discover

# 4. Safe read-only query smoke test
dataagent query "SELECT COUNT(*) FROM \"ecommerce\".\"public\".\"orders\" LIMIT 1"

# 5. Generate candidate question/SQL pairs, validate, verify, export
#    (with GLM_API_KEY set and GLM_MODE=zai; or GLM_MODE=mock offline)
dataagent generate --target 500
```

See `docs/SETUP.md` and `docs/FINETUNING.md`. A sample verified dataset
(offline template mode) ships in `data/sample/`.

## Security

- Credentials in environment variables / secrets manager only; never logged.
- All generated-SQL execution uses a dedicated **read-only** Dremio account.
- Write/DML statements rejected at parse level (sqlglot) *and* blocked by the
  account's privileges (defense in depth).
- Execution timeouts + result-row limits; audit log of generated/executed SQL.
- Candidate SQL execution is opt-in (`ENABLE_QUERY_EXECUTION`).
- Only schema metadata and approved descriptions are sent to the LLM — never
  confidential query results.

## Fine-tuning

`docs/FINETUNING.md` contains instructions for SFT of Qwen3.5-0.8B with
LLaMA-Factory using the exported `messages` JSONL, plus evaluation on the
held-out test split.
