"""OpenCode Go model client (OpenAI-compatible Console inference API).

Verified provider facts (see https://opencode.ai/console/go and
https://opencode.ai/v2/docs/console/api/inference/):
  - Go subscription base URL:
      https://opencode.ai/inference/go/openai/v1/chat/completions
  - Model IDs served via Go (OpenAI Chat Completions family):
      glm-5.3-flash, glm-5.3, glm-5.2, deepseek-v4.1-flash, kimi-k3, ...
  - Live model list: https://opencode.ai/inference/go/v1/models
  - Auth: `Authorization: Bearer <console API key>` (personal key `oc_sk_...`
    or service-account key; Go usage counts against the subscription).

Credential resolution order (nothing is logged or copied into the repo):
  1. OPENCODE_API_KEY environment variable
  2. OpenCode's own auth store (~/.local/share/opencode/auth.json,
     provider "opencode-go")
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from dataagent.config import Settings, get_settings
from dataagent.llm.base import ChatMessage, ModelResponse
from dataagent.logging_utils import get_logger

log = get_logger(__name__)

OPENCODE_AUTH_FILE = Path.home() / ".local" / "share" / "opencode" / "auth.json"


class OpenCodeError(Exception):
    pass


def resolve_opencode_key(settings: Settings | None = None) -> str:
    import os

    # 1. Explicit setting (e.g. OPENCODE_API_KEY in .env, read by pydantic-settings)
    if settings is not None and settings.opencode_api_key:
        return settings.opencode_api_key.strip()
    # 2. Process environment variable
    key = os.getenv("OPENCODE_API_KEY", "").strip()
    if key:
        return key
    # 3. OpenCode's own auth store
    if OPENCODE_AUTH_FILE.exists():
        try:
            data = json.loads(OPENCODE_AUTH_FILE.read_text())
            entry = data.get("opencode-go")
            if isinstance(entry, dict) and entry.get("key"):
                return str(entry["key"]).strip()
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read opencode auth store: %s", exc)
    raise OpenCodeError(
        "No OpenCode API key found; set OPENCODE_API_KEY in .env, export it, "
        "or connect opencode with `/connect` (provider opencode-go)"
    )


class OpenCodeModelClient:
    """OpenAI-compatible chat client for the OpenCode Console/Go gateway."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._key = resolve_opencode_key(self._settings)
        base = self._settings.opencode_base_url.rstrip("/")
        # Model list lives at .../go/v1/models (not under /openai/v1).
        self._models_url = base.replace("/openai/v1", "/v1") + "/models"
        # Go requires a stable per-conversation routing header; one UUID per
        # client instance keeps a pipeline run on the same route/prompt cache.
        self._session_id = str(uuid.uuid4())
        self._http = httpx.Client(
            base_url=base,
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json",
                     "x-opencode-session": self._session_id},
            timeout=300.0,
        )

    @property
    def model_id(self) -> str:
        return f"opencode-go/{self._settings.glm_model}"

    def close(self) -> None:
        self._http.close()

    # -- provider verification --------------------------------------------------

    def verify_provider(self) -> list[str]:
        """Fetch the live model list and confirm the configured model exists."""
        resp = self._http.get(self._models_url)
        if resp.status_code == 401:
            raise OpenCodeError("OpenCode API key rejected (401)")
        if resp.status_code >= 400:
            raise OpenCodeError(f"Model list failed {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        ids = [m.get("id", "") for m in payload.get("data", [])]
        model = self._settings.glm_model
        if ids and model not in ids:
            raise OpenCodeError(
                f"Model '{model}' not served by OpenCode Go; available GLM models: "
                + ", ".join(m for m in ids if m.startswith("glm")) or "none")
        log.info("Verified OpenCode Go serves model %s", model)
        return ids

    # -- chat ---------------------------------------------------------------------

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> ModelResponse:
        body: dict[str, Any] = {
            "model": self._settings.glm_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # GLM models always reason; effort is configurable on the gateway.
            "reasoning_effort": self._settings.glm_reasoning_effort,
        }
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = self._http.post("/chat/completions", json=body)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    log.warning("Rate-limited by OpenCode; retrying in %ss", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code in (401, 402, 403):
                    raise OpenCodeError(
                        f"OpenCode auth/usage error {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    raise OpenCodeError(f"OpenCode API {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                message = data["choices"][0]["message"]
                content = message.get("content")
                if content is None:
                    # reasoning consumed the whole budget: retry with more tokens
                    if body["max_tokens"] < 32768:
                        body["max_tokens"] = min(body["max_tokens"] * 4, 32768)
                        log.info("Model returned reasoning-only output; retrying with max_tokens=%s",
                                 body["max_tokens"])
                        continue
                    raise OpenCodeError("Model exhausted max_tokens on reasoning; no content")
                return ModelResponse(content=content, model=self.model_id, raw=data)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                wait = 2 ** attempt
                log.warning("OpenCode call failed (%s); retrying in %ss", exc, wait)
                time.sleep(wait)
        raise OpenCodeError(f"OpenCode API unusable after {retries} attempts: {last_error}")
