"""Question triage: classify each question BEFORE SQL generation.

Verdicts:
  ready               — specific enough to generate SQL
  needs_clarification — multiple materially different interpretations
  unsupported         — the available schema cannot answer it reliably

Only `ready` questions proceed to SQL generation. Clarification examples are
exported separately (opt-in) so the fine-tuned model's contract stays clean
unless clarification handling is intended at inference time.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from dataagent.generation.prompts import SQL_SYSTEM, QUESTION_SYSTEM  # noqa: F401
from dataagent.generation.questions import schema_text
from dataagent.llm.base import ChatMessage, ModelClient, extract_json
from dataagent.logging_utils import get_logger

log = get_logger(__name__)

TRIAGE_SYSTEM = """\
You triage analytical questions against a Dremio schema. Decide whether the \
question is specific enough to translate into one unambiguous SQL query.

Verdicts:
- "ready": the question names its tables/rows/metric/filter clearly enough \
that one query answers it.
- "needs_clarification": multiple materially different interpretations exist \
(e.g. "show me the sales figures" without saying which metric or grouping), \
or a requested metric has no definition in the schema.
- "unsupported": the schema cannot answer the question at all (missing data, \
forecasting, sentiment, external sources).

Be strict: when a metric is named but ambiguous ("revenue" with no \
definition, "performance", "satisfaction"), prefer needs_clarification. \
Return ONLY JSON.\
"""

TRIAGE_USER_TEMPLATE = """\
Schema:
{schema_text}

Question: {question}
Category hint: {category}

Decide: ready | needs_clarification | unsupported.
If needs_clarification, list the materially different interpretations.

Return JSON:
{{"verdict": "ready"|"needs_clarification"|"unsupported",
  "reason": "<short explanation>",
  "interpretations": ["..."],   // only for needs_clarification
  "confidence": 0.0-1.0}}\
"""

AMBIGUITY_HINTS = re.compile(
    r"\b(sales figures|performed best|performance|satisfaction|grew|popular|"
    r"busiest|most important|best|top selling|engagement|activity|interest)\b",
    re.IGNORECASE,
)


class TriageService:
    def __init__(self, model: ModelClient | None = None) -> None:
        self._model = model

    def classify(self, question: str, category: str, schema_block: str) -> dict:
        verdict, reason, interpretations = self._rule_based(question, category)
        if verdict is None and self._model is not None:
            verdict, reason, interpretations = self._llm_based(
                question, category, schema_block)
        return {"verdict": verdict or "needs_clarification",
                "reason": reason, "interpretations": interpretations}

    # -- deterministic rules first (cheap, reproducible) ---------------------

    def _rule_based(self, question, category):
        # unsupported categories are explicit in our taxonomy
        if category == "unsupported":
            return "unsupported", "category marked unsupported (schema cannot answer)", []
        if category in ("schema_ambiguity", "missing_info"):
            return "needs_clarification", (
                f"category {category}: the question under-specifies its target"), []
        # generic ambiguity words without metric/table qualification
        if AMBIGUITY_HINTS.search(question) and not re.search(
                r"\b(amount|revenue|count|number|quantity|sales|price|cost|"
                r"\w+_id|name|email|region|tier|status)\b\s",
                question, re.IGNORECASE):
            return None, "", []  # let the LLM decide with schema context
        return None, "", []

    # -- LLM triage -------------------------------------------------------------

    def _llm_based(self, question, category, schema_block):
        user = TRIAGE_USER_TEMPLATE.format(
            schema_text=schema_block, question=question, category=category)
        resp = self._model.complete(
            [ChatMessage("system", TRIAGE_SYSTEM), ChatMessage("user", user)],
            temperature=0.1, max_tokens=2048)
        try:
            payload = extract_json(resp.content)
        except ValueError:
            return "needs_clarification", "triage output not JSON", []
        if not isinstance(payload, dict) or "verdict" not in payload:
            return "needs_clarification", "triage output malformed", []
        verdict = payload.get("verdict")
        if verdict not in ("ready", "needs_clarification", "unsupported"):
            verdict = "needs_clarification"
        return verdict, str(payload.get("reason", "")), \
            list(payload.get("interpretations") or [])


class RuleBasedTriage(TriageService):
    """Deterministic-only triage for mock mode (no model calls)."""

    def classify(self, question, category, schema_block):
        verdict, reason, _ = self._rule_based(question, category)
        if verdict is None:
            verdict, reason = "ready", "rule-based triage (mock mode)"
        return {"verdict": verdict, "reason": reason, "interpretations": []}


def triage_all(service: TriageService, questions, schema_block: str,
               workers: int = 4) -> None:
    """Annotate each CandidateQuestion in place with triage_verdict."""
    def run(q):
        result = service.classify(q.question, q.category, schema_block)
        q.triage_verdict = result["verdict"]
        q.triage_reason = result["reason"]
        q.triage_interpretations = result["interpretations"]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(run, questions))
