"""Dremio error classification and the GLM correction loop.

Retries are capped; permission/authentication failures are never retried with
a rewritten query (bypassing access controls is forbidden).
"""

from __future__ import annotations

import re

from dataagent.config import Settings, get_settings
from dataagent.dremio.client import (
    AuthenticationError,
    DremioError,
    PermissionError,
)
from dataagent.generation.prompts import CORRECTION_TEMPLATE, QUESTION_SYSTEM
from dataagent.llm.base import ChatMessage, ModelClient
from dataagent.logging_utils import get_logger

log = get_logger(__name__)

SQL_PATTERNS = {
    "syntax_error": re.compile(r"(INVALID SQL|parse|syntax|not a valid)", re.IGNORECASE),
    "object_not_found": re.compile(r"(not found|does not exist|unknown|INVALID_OBJECT|no schema)", re.IGNORECASE),
    "type_mismatch": re.compile(r"(type|aggregation|cannot be cast|incompatible)", re.IGNORECASE),
    "permission": re.compile(r"(permission|access denied|not authorized|unauthorized)", re.IGNORECASE),
    "permission": re.compile(r"(permission|access denied|not authorized|unauthorized)", re.IGNORECASE),
    "source_unavailable": re.compile(r"(source.*(error|unavailable)|table.*offline|connection)", re.IGNORECASE),
    "timeout": re.compile(r"(timeout|timed out)", re.IGNORECASE),
}


def classify_error(message: str) -> str:
    if not message:
        return "unknown"
    for name, pattern in SQL_PATTERNS.items():
        if pattern.search(message):
            return name
    return "sql_error"


NON_SQL_CLASSES = {"authentication", "permission", "timeout", "source_unavailable"}


class CorrectionEngine:
    """Ask GLM to correct a failing query, then revalidate (caller re-runs)."""

    def __init__(self, model: ModelClient, settings: Settings | None = None,
                 max_retries: int | None = None) -> None:
        self._model = model
        self._settings = settings or get_settings()
        self._max_retries = max_retries or self._settings.max_correction_retries

    def correct(self, question: str, sql: str, error_message: str,
                schema_text: str) -> str | None:
        error_class = classify_error(error_message)
        if error_class in NON_SQL_CLASSES:
            log.info("Skipping correction for non-SQL error class %s", error_class)
            return None
        user = CORRECTION_TEMPLATE.format(
            question=question, sql=sql, error=error_message[:800],
            error_class=error_class, schema_text=schema_text,
        )
        resp = self._model.complete([ChatMessage("system", QUESTION_SYSTEM), ChatMessage("user", user)],
                                    temperature=0.2, max_tokens=2048)
        content = resp.content.strip()
        # keep only SQL text
        lines = [l for l in content.splitlines() if l.strip()]
        candidate = "\n".join(lines).strip()
        if candidate.lower().startswith(("```sql", "```")):
            candidate = candidate.strip("`").replace("sql\n", "").strip()
        return candidate
