"""Environment-driven settings for dataagent."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Overridable so tests/demos can use an isolated data directory.
DATA_DIR = Path(os.getenv("DATAAGENT_DATA_DIR", str(PROJECT_ROOT / "data")))
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"
AUDIT_DIR = OUTPUT_DIR / "audit"
SCHEMA_CACHE_DB = DATA_DIR / "metadata_cache.sqlite3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Dremio
    dremio_uri: str = "http://localhost:9047"
    dremio_user: str = "dataagent_ro"
    dremio_password: str = ""
    dremio_pat: str = ""
    # Admin credentials are used *only* by scripts/bootstrap_dremio.py.
    dremio_admin_user: str = "dremio_admin"
    dremio_admin_password: str = ""

    # GLM provider (Z.AI / Zhipu OpenAI-compatible endpoint)
    glm_base_url: str = "https://api.z.ai/api/paas/v4"
    glm_api_key: str = ""
    glm_model: str = "glm-5.3"
    glm_reasoning_effort: str = "high"
    glm_mode: str = "zai"  # zai | opencode | mock

    # OpenCode Go provider (mode=opencode): Console inference API
    opencode_base_url: str = "https://opencode.ai/inference/go/openai/v1"
    # API key resolved from OPENCODE_API_KEY or opencode's auth store; keep empty
    opencode_api_key: str = ""

    # Pipeline controls
    enable_query_execution: bool = True
    query_timeout_seconds: int = 60
    query_max_rows: int = 100
    generation_batch_size: int = 25
    max_correction_retries: int = 2
    dataset_version: str = "0.1.0"
    # Thread pool size for parallel SQL generation / execution / review.
    pipeline_workers: int = 4
    # Opt-in embedding-based near-duplicate detection (requires chromadb).
    # Calibration (2026-10): MiniLM/potion embeddings cannot reliably separate
    # paraphrases from "same shape, different metric" questions on short
    # text-to-SQL questions (gap ~0.03) — keep off unless re-evaluated.
    semantic_dedup_enabled: bool = False
    semantic_similarity_threshold: float = 0.85
    # Export clarification-style examples (question -> clarifying reply) as a
    # separate split; default off to keep the SFT contract SQL-only.
    export_clarifications: bool = False
    # Estimated provider pricing (USD per 1M tokens) for cost reporting.
    llm_input_price_per_m: float = 0.15
    llm_output_price_per_m: float = 0.50


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return settings
