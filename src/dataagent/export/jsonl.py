"""JSONL exporter.

Produces:
  - <name>.jsonl  : one object per line, {"messages":[system,user,assistant]}
  - manifest.jsonl : example_id + full QC metadata (never mixed into the SFT
                     content — the assistant target is SQL text only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from dataagent.config import OUTPUT_DIR, get_settings
from dataagent.logging_utils import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "Generate valid Dremio SQL using only the provided schema. "
    "Never invent tables or columns. Use read-only SELECT/WITH statements only."
)


def build_user_content(schema_block: str, question: str) -> str:
    return f"Schema: {schema_block}\nQuestion: {question}"


def render_example(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["user_content"]},
            {"role": "assistant", "content": example["assistant_sql"]},
        ]
    }


class JSONLExporter:
    def __init__(self, out_dir: Path | None = None, version: str | None = None) -> None:
        self._dir = out_dir or OUTPUT_DIR
        self._version = version or get_settings().dataset_version

    def write_split(self, split: str, examples: list[dict]) -> tuple[Path, Path]:
        """Write train/validation/test SFT file + one shared manifest line each."""
        sft_path = self._dir / f"{split}.jsonl"
        manifest_path = self._dir / f"{split}.manifest.jsonl"
        with sft_path.open("w", encoding="utf-8") as sft, \
                manifest_path.open("w", encoding="utf-8") as man:
            for example in examples:
                rendered = (self._render_clarification(example)
                            if split == "clarifications"
                            else render_example(example))
                sft.write(json.dumps(rendered, ensure_ascii=False) + "\n")
                man.write(json.dumps(self._manifest_entry(example, split),
                                     ensure_ascii=False) + "\n")
        return sft_path, manifest_path

    @staticmethod
    def _render_clarification(example: dict) -> dict:
        """Clarification target: the assistant asks, it never guesses."""
        interps = example.get("interpretations") or []
        bullet = "; ".join(f"({i+1}) {t}" for i, t in enumerate(interps[:4]))
        reply = ("I need clarification before I can answer. "
                 + (f"Did you mean: {bullet}? " if bullet else "")
                 + "Please specify the metric, grouping and period you want.")
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": reply},
        ]}

    def _manifest_entry(self, example: dict, split: str = "") -> dict[str, Any]:
        semantic = example.get("semantic_status", "skipped")
        execution = example.get("execution_status", "not_executed")
        if semantic == "accept" and execution == "completed":
            review_status = "verified"
        elif semantic == "needs_review" or execution in ("failed", "timed_out"):
            review_status = "pending"
        else:
            review_status = "unverified"
        return {
            "example_id": example.get("example_id", ""),
            "schema_group": example.get("schema_group"),
            "category": example.get("category"),
            "difficulty": example.get("difficulty"),
            "candidate_sql": example.get("candidate_sql", example.get("assistant_sql")),
            "sql_fingerprint": example.get("sql_fingerprint"),
            "validation_status": example.get("validation_status"),
            "schema_validation_status": example.get("validation_status"),
            "execution_status": execution,
            "semantic_status": semantic,
            "semantic_findings": example.get("semantic_findings", []),
            "reference_comparison": example.get("reference_comparison", "not_applicable"),
            "review_status": review_status,
            "acceptance_status": "accepted"
            if review_status == "verified" and split in ("train", "validation", "test")
            else split or "rejected",
            "rejection_reason": example.get("rejection_reason"),
            "triage_verdict": example.get("triage_verdict", "ready"),
            "triage_reason": example.get("triage_reason", ""),
            "generation_model": example.get("generation_model"),
            "dataset_version": self._version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
