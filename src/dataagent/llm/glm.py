"""GLM model client (Z.AI / Zhipu OpenAI-compatible chat completion API).

Verified provider facts (DO NOT change without checking https://docs.z.ai):
  - Model ID:          glm-5.3 (flagship) or glm-5.3-flash (lower cost)
  - Base URL intl:     https://api.z.ai/api/paas/v4
  - Base URL CN:       https://open.bigmodel.cn/api/paas/v4
  - Auth:              Bearer token in Authorization header
  - Reasoning:         glm-5.3 always reasons; thinking.type must be "enabled";
                       reasoning_effort in {low, high, max}
All values come from configuration (GLM_BASE_URL / GLM_API_KEY / GLM_MODEL /
GLM_REASONING_EFFORT) so the provider can be swapped at any time.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from dataagent.config import Settings, get_settings
from dataagent.llm.base import ChatMessage, ModelResponse
from dataagent.logging_utils import get_logger

log = get_logger(__name__)


class LLMError(Exception):
    pass


class GLMClient:
    """OpenAI-compatible chat client for Z.AI GLM models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.glm_api_key:
            raise LLMError(
                "GLM_API_KEY is empty; configure .env (or set GLM_MODE=mock in settings "
                "to use MockModelClient for development)"
            )
        self._http = httpx.Client(
            base_url=self._settings.glm_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self._settings.glm_api_key}",
                "Content-Type": "application/json",
            },
            timeout=300.0,
        )

    @property
    def model_id(self) -> str:
        return f"glm/{self._settings.glm_model}"

    def close(self) -> None:
        self._http.close()

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
            "thinking": {"type": "enabled"},
            "reasoning_effort": self._settings.glm_reasoning_effort,
        }
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = self._http.post("/chat/completions", json=body)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    log.warning("Rate-limited by GLM API; retrying in %ss", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    raise LLMError(f"GLM API {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return ModelResponse(content=content, model=self.model_id, raw=data)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                wait = 2 ** attempt
                log.warning("GLM call failed (%s); retrying in %ss", exc, wait)
                time.sleep(wait)
        raise LLMError(f"GLM API unusable after {retries} attempts: {last_error}")


class MockModelClient:
    """Deterministic mock client for development/tests (no network, no key)."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._cursor = 0

    @property
    def model_id(self) -> str:
        return "mock/mock-model"

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.7,
                 max_tokens: int = 4096) -> ModelResponse:
        if self._cursor < len(self._responses):
            content = self._responses[self._cursor]
            self._cursor += 1
        else:
            content = "{}"
        return ModelResponse(content=content, model=self.model_id, raw={})


def build_model_client(settings: Settings | None = None):
    """Provider factory — the single point where model code is chosen."""
    mode = (settings or get_settings()).glm_mode
    if mode == "mock":
        return MockModelClient()
    if mode == "opencode":
        from dataagent.llm.opencode import OpenCodeModelClient

        return OpenCodeModelClient(settings)
    return GLMClient(settings)
