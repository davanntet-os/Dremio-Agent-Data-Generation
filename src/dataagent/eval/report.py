"""Dataset quality / acceptance / rejection reporting with cost analysis."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from dataagent.config import REPORTS_DIR
from dataagent.logging_utils import get_logger

log = get_logger(__name__)

STAGE_BUCKETS = {
    "invalid_json": "Invalid JSON count",
    "json_schema": "JSON-Schema rejection count",
}

RECOMMENDATIONS = {
    "semantic_reject": (
        "Many candidates were rejected by the semantic reviewer: sharpen the "
        "SQL-generation prompt with metric/filter definitions, or restrict "
        "generation to categories with better acceptance."),
    "failed_execution": (
        "Frequent execution failures: add schema-aware few-shot examples and "
        "verify SQL function syntax against the target Dremio version."),
    "missing_requested_limit": (
        "Questions request top-N results but generated SQL omits LIMIT: "
        "enforce LIMIT-when-requested in the generation prompt."),
    "missing_identifier": (
        "Detail listings omit identifying columns: add the identifier "
        "requirement to the generation prompt."),
    "needs_clarification": (
        "Ambiguous questions are triaged, not guessed: consider exporting "
        "clarification examples (EXPORT_CLARIFICATIONS=true) if the deployed "
        "model should ask instead of guess."),
    "unsupported": (
        "Unsupported questions were rejected rather than answered with "
        "invented schema: expected behavior."),
    "duplicate_sql": (
        "Duplicate SQL fingerprints: increase generation temperature or "
        "diversify question phrasing."),
    "duplicate_question": (
        "Near-duplicate questions detected: raise the dedup strictness or "
        "prompt for phrasing variety."),
}


class ReportWriter:
    def __init__(self, reports_dir: Path | None = None, version: str = "0.1.0") -> None:
        self._dir = reports_dir or REPORTS_DIR
        self._version = version

    def write(self, summary: dict) -> Path:
        path = self._dir / f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        payload = {"dataset_version": self._version, "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **summary}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        log.info("Report written to %s", path)
        return path

    @staticmethod
    def summarize(candidate_questions, accepted: list[dict],
                  rejected: dict[str, list[str]],
                  split_counts: dict[str, int]) -> dict:
        by_category = Counter(e.get("category") for e in accepted)
        by_difficulty = Counter(e.get("difficulty") for e in accepted)

        def bucket(*names: str) -> int:
            return sum(len(rejected.get(n, [])) for n in names)

        stage_counts = {
            "invalid_json": bucket("invalid_json", "model returned unparseable JSON"),
            "json_schema_rejections": bucket("json_schema"),
            "sql_parse_failures": bucket("SQL parse error", "Forbidden keyword present (regex pre-check)",
                                         "Statement type not allowed: SelectUnknown",
                                         "SELECT * requires an explicit LIMIT <= max rows"),
            "schema_validation_failures": bucket("Unknown table", "Unknown column", "invented"),
            "execution_failures": bucket("failed_execution", "sql_error", "dremio"),
            "semantic_mismatches": bucket("semantic_reject", "semantic_reject_review"),
            "reference_mismatches": bucket("reference_mismatch"),
            "ambiguous_needing_clarification": bucket("needs_clarification"),
            "unsupported": bucket("unsupported"),
            "duplicates": bucket("duplicate_sql", "duplicate_question", "duplicate_across_runs"),
            "missing_requested_limit": bucket("missing_requested_limit"),
        }
        accepted_verified = len([e for e in accepted if e.get("review_status", "verified") == "verified"])
        recommendations = {}
        for name, text in RECOMMENDATIONS.items():
            if rejected.get(name):
                recommendations[name] = text
        top_failures = sorted(rejected.items(), key=lambda kv: -len(kv[1]))[:5]
        return {
            "candidates": len(candidate_questions),
            "accepted": len(accepted),
            "accepted_verified": accepted_verified,
            "accepted_ratio": round(len(accepted) / max(1, len(candidate_questions)), 3),
            "rejected": {reason: {"count": len(ids)} for reason, ids in rejected.items()},
            "top_failure_patterns": [{"reason": r, "count": len(ids)} for r, ids in top_failures],
            "stage_counts": stage_counts,
            "recommendations": recommendations,
            "by_category": dict(by_category),
            "by_difficulty": dict(by_difficulty),
            "splits": split_counts,
        }
