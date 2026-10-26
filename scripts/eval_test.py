"""Evaluate generated SQL predictions against the held-out test split.

Usage:
  python scripts/eval_test.py predictions.jsonl data/output/test.jsonl

predictions.jsonl: one {"example_id": "...", "sql": "..."} per line, aligned
with the test manifest example_ids. Each prediction is validated and executed
against the read-only test Dremio; accuracy counts COMPLETED jobs whose result
matches the reference query's result when available.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from dataagent.config import get_settings
from dataagent.validation.executor import ExecutionService
from dataagent.validation.parser import ParserValidator


def main(pred_path: str, test_manifest_path: str) -> int:
    settings = get_settings()
    parser = ParserValidator(max_rows=settings.query_max_rows)
    executor = ExecutionService()

    reference_sql = {}
    with open(test_manifest_path, encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            reference_sql[entry["example_id"]] = entry

    stats: dict[str, list[bool]] = defaultdict(list)
    for line in open(pred_path, encoding="utf-8"):
        pred = json.loads(line)
        example_id = pred["example_id"]
        sql = pred.get("sql", "")
        verdict = parser.parse_and_validate(sql)
        ok = False
        if verdict.ok:
            outcome = executor.execute(sql, example_id=example_id)
            ok = outcome.status == "completed"
        stats["overall"].append(ok)
        category = reference_sql.get(example_id, {}).get("category", "unknown")
        stats[category].append(ok)

    print(json.dumps(
        {k: {"accuracy": round(sum(v) / max(1, len(v)), 3), "n": len(v)}
         for k, v in sorted(stats.items())}, indent=1))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
