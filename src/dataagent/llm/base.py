"""Model-client abstraction.

The Dremio integration never imports model code directly; providers live in
dataagent.llm.* and are selected via configuration so the model can be swapped
without touching the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ModelResponse:
    content: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


class UsageTracker:
    """Thread-safe accumulation of token usage from model calls."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add_usage(self, usage: dict[str, Any]) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens}

    def estimated_cost(self, input_per_m: float, output_per_m: float) -> float:
        with self._lock:
            p, c = self.prompt_tokens, self.completion_tokens
        return (p / 1_000_000) * input_per_m + (c / 1_000_000) * output_per_m


class TrackedModelClient:
    """Wraps any ModelClient and records token usage."""

    def __init__(self, inner: ModelClient, tracker: UsageTracker) -> None:
        self._inner = inner
        self._tracker = tracker

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.7,
                 max_tokens: int = 4096) -> ModelResponse:
        resp = self._inner.complete(messages, temperature=temperature,
                                    max_tokens=max_tokens)
        self._tracker.add_usage(resp.raw.get("usage", {}) if resp.raw else {})
        return resp


class ModelClient(Protocol):
    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.7,
                 max_tokens: int = 4096) -> ModelResponse: ...

    @property
    def model_id(self) -> str: ...


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array from a model response.

    Handles fenced ```json blocks and raw JSON. Raises ValueError when no
    parseable JSON is present.
    """
    candidates: list[str] = []
    for match in _JSON_FENCE.finditer(text):
        candidates.append(match.group(1))
    spans = _balanced_spans(text)
    # prefer the largest span (outermost structure) over nested fragments
    candidates.extend(sorted(spans, key=len, reverse=True))
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("No parseable JSON in model response")


def _balanced_spans(text: str) -> list[str]:
    spans = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for idx in range(start, min(len(text), start + 200_000)):
                ch = text[idx]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        spans.append(text[start:idx + 1])
                        break
            start = text.find(open_ch, start + 1)
    return spans
