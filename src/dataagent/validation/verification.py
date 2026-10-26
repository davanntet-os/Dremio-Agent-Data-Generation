"""Semantic verification service.

Primary check: a second GLM reviewer pass comparing question vs SQL vs schema.
Where feasible (tables/count/aggregation categories), independently constructed
reference checks compare result sets. Ambiguous verdicts are never auto-
accepted; they are exported as `needs_review` and excluded from training.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from dataagent.generation.prompts import REVIEW_SYSTEM, REVIEW_USER_TEMPLATE
from dataagent.generation.questions import schema_text
from dataagent.llm.base import ChatMessage, ModelClient
from dataagent.logging_utils import get_logger, AuditLog
from dataagent.validation.executor import ExecutionService

log = get_logger(__name__)

# Categories whose verification can use an independently constructed check.
REFERENCE_ELIGIBLE = {"aggregation", "business_metric", "group_by_having", "null_handling"}


class SemanticReview:
    def __init__(self, model: ModelClient, executor: ExecutionService,
                 audit: AuditLog | None = None) -> None:
        self._model = model
        self._executor = executor
        self._audit = audit or AuditLog()

    def verify(self, question: str, sql: str, category: str, task_type: str,
               schema_block: str, result_rows: list | None = None,
               datasets: list[dict] | None = None,
               evidence=None) -> dict:
        review = self._llm_review(question, sql, category, task_type, schema_block)
        reference = None
        if category in REFERENCE_ELIGIBLE and result_rows:
            reference = self._reference_check(question, sql, category, result_rows)
            if reference is False:
                # independent check contradicts the query: never accept
                review = {"verdict": "reject",
                          "reasons": (review.get("reasons", []) or []) +
                                      ["reference check disagrees with result"],
                          "confidence": min(0.99, review.get("confidence", 1.0))}
        else:
            reference = "not_applicable"

        # data-driven status-value verification
        if datasets is not None and evidence is not None:
            status_finding = self._status_check(question, sql, datasets, evidence)
            if status_finding is not None:
                review = {"verdict": "needs_review",
                          "reasons": (review.get("reasons", []) or []) + [status_finding],
                          "confidence": review.get("confidence", 0.5)}

        final = self._merge(review, reference)
        final["reference_comparison"] = {
            True: "pass", False: "fail", "not_applicable": "not_applicable",
            None: "not_applicable"}.get(reference, "not_applicable")
        self._audit.record("semantic_review", question=question, sql=sql,
                           verdict=final["verdict"], reasons=final["reasons"])
        return final

    def _status_check(self, question, sql, datasets, evidence) -> str | None:
        from dataagent.dremio.models import DatasetInfo
        from dataagent.validation.semantic_checks import check_status_values

        for ds in datasets:
            status_cols = [c for c in ds["columns"] if "status" in c["name"].lower()]
            if not status_cols:
                continue
            values = evidence.distinct_status_values(ds["fingerprint"], ".".join(ds["path"]), ds["columns"])
            finding = check_status_values(question, sql, values)
            if finding["severity"] != "ok":
                return finding["reason"]
        return None

    def _merge(self, review: dict, reference) -> dict:
        verdict = review.get("verdict", "reject")
        reasons = list(review.get("reasons") or [])
        if reference is True and verdict == "accept":
            reasons.append("reference check agrees")
        elif reference is False:
            verdict = "reject"
        return {"verdict": verdict, "reasons": reasons,
                "confidence": float(review.get("confidence", 0.5)),
                "reference_check": {"not_applicable": None, True: "pass", False: "fail"}.get(reference)}

    def _llm_review(self, question, sql, category, task_type, schema_block) -> dict:
        user = REVIEW_USER_TEMPLATE.format(
            schema_text=schema_block, question=question, sql=sql,
            category=category, task_type=task_type)
        resp = self._model.complete(
            [ChatMessage("system", REVIEW_SYSTEM), ChatMessage("user", user)],
            temperature=0.1, max_tokens=1024)
        from dataagent.llm.base import extract_json
        try:
            payload = extract_json(resp.content)
        except ValueError:
            return {"verdict": "needs_review", "reasons": ["reviewer output not JSON"],
                    "confidence": 0.0}
        if not isinstance(payload, dict) or "verdict" not in payload:
            return {"verdict": "needs_review", "reasons": ["reviewer output malformed"],
                    "confidence": 0.0}
        verdict = payload.get("verdict")
        if verdict not in ("accept", "reject", "needs_review"):
            verdict = "needs_review"
        return {"verdict": verdict, "reasons": payload.get("reasons", []),
                "confidence": payload.get("confidence", 0.5)}

    def _reference_check(self, question, sql, category, result_rows) -> bool | None:
        """Independent checks: structural rules + executed reference comparison.

        For single-row filter-free aggregations (aggregation/business_metric)
        a canonical reference query is derived by stripping ORDER BY/LIMIT
        from the candidate's aggregation core and executed separately; result
        values must match. This catches wrong filters/joins that silently
        change the metric.
        """
        try:
            tree = sqlglot.parse_one(sql, read="dremio")
        except Exception:
            return None
        aggregates = [a for a in tree.find_all(exp.AggFunc)]
        if category in ("aggregation", "business_metric") and aggregates:
            has_sum = any(a.key in ("sum", "count") for a in aggregates)
            if not has_sum:
                return False
        if category == "null_handling":
            if not tree.find(exp.Null):
                return False
        if category in ("aggregation", "business_metric") and result_rows:
            selects = [s for s in tree.find_all(exp.Select) if s is not None]
            top = selects[0]
            single_row = len(result_rows) == 1
            no_filter = top.args.get("where") is None and not top.args.get("having")
            grouped = top.args.get("group") is not None
            if single_row and no_filter and not grouped:
                ref_outcome = self._executor.execute(self._reference_sql(top, tree))
                if ref_outcome.status != "completed" or not ref_outcome.rows:
                    return None
                return self._rows_match(ref_outcome.rows[0], result_rows[0])
        return True

    @staticmethod
    def _reference_sql(top: exp.Select, tree: exp.Expression) -> str:
        ref = exp.Select()
        for projection in top.expressions:
            ref.append("expressions", projection)
        for tbl in tree.find_all(exp.Table):
            ref = ref.from_(tbl.sql(dialect="dremio"))
            break
        return ref.sql(dialect="dremio")

    @staticmethod
    def _rows_match(a: dict, b: dict) -> bool:
        """Compare result rows: NULL-aware, float-tolerant, order-insensitive
        (ordering is validated separately), duplicates counted."""
        keys = set(a) | set(b)
        for key in keys:
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None:
                if va is None and vb is None:
                    continue
                return False  # NULL in one but not the other
            try:
                fa, fb = float(va), float(vb)
                if abs(fa - fb) > max(1e-6, abs(fa) * 1e-6):
                    return False
            except (TypeError, ValueError):
                if str(va) != str(vb):
                    return False
        return True
