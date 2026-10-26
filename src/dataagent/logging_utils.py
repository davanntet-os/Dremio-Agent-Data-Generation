"""Audit/standard logging helpers. Never log API keys or tokens."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from dataagent.config import AUDIT_DIR

_SENSITIVE_KEYS = {"password", "token", "pat", "api_key", "authorization", "secret"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def redact(obj: Any) -> Any:
    """Recursively remove sensitive values from payload-like structures."""
    if isinstance(obj, dict):
        return {
            k: ("<redacted>" if k.lower() in _SENSITIVE_KEYS else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


class AuditLog:
    """Append-only JSONL audit log of generated and executed queries."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or AUDIT_DIR / "audit.jsonl"
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: str, **details: Any) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "entry_id": str(uuid.uuid4()),
            **redact(details),
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
