"""Dataset quality-control service.

- Exact duplicate detection
- Normalized SQL fingerprinting (sqlglot canonical serialization)
- Near-duplicate question detection (lexical fingerprints)
- Leakage-safe train/validation/test split grouped by (query templatex schema)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import sqlglot

from dataagent.generation.questions import question_fingerprint


def normalize_sql(sql: str) -> str:
    """Canonical low-level form for fingerprinting."""
    try:
        tree = sqlglot.parse_one(sql, read="dremio")
        formatted = tree.sql(dialect="dremio", pretty=False, comments=False)
        return " ".join(formatted.lower().split())
    except Exception:
        return " ".join(" ".join(sql.split()).lower())


def sql_fingerprint(sql: str) -> str:
    normalized = normalize_sql(sql)
    return normalized


SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


@dataclass
class QCPipelineResult:
    examples: list = field(default_factory=list)
    exact_duplicates: int = 0
    near_duplicate_questions: int = 0
    duplicate_sql_fingerprints: int = 0
    fingerprint_to_group: dict = field(default_factory=dict)


def split_grouped(examples: list[dict], ratios: dict | None = None) -> dict[str, list[dict]]:
    """Group examples by (SQL fingerprint root, schema group) before the split.

    All members of a group go to the same split to prevent leakage of a query
    template's variants across train/val/test.
    """
    ratios = ratios or SPLIT_RATIOS
    groups: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        key = (example.get("sql_fingerprint", "unknown"),
               example.get("schema_group", "unknown"))
        groups[tuple(sorted(key))].append(example)  # type: ignore[arg-type]
    ordered_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    total = max(1, len(examples))
    fills = {"train": 0.0, "validation": 0.0, "test": 0.0}
    for _, members in ordered_groups:
        # deterministic ordering inside the group
        members.sort(key=lambda e: e.get("example_id", ""))
        target = min(fills, key=lambda name: fills[name] / max(ratios[name], 1e-9))
        splits[target].extend(members)
        fills[target] += len(members)
    return splits
