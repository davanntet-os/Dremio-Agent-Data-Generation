"""SQL-generation engine.

- SQLGenerationEngine: GLM path, output validated against SQL_OUTPUT_SCHEMA
  (validation stage 1) before anything else touches the candidate.
- TemplateSQLGenerator: deterministic producer for mock/dev mode that writes
  real Dremio SQL for each template question category.
"""

from __future__ import annotations

import json

import jsonschema
from sqlglot import exp
import sqlglot

from dataagent.generation.models import SQLCandidate, SQL_OUTPUT_SCHEMA, CandidateQuestion
from dataagent.generation.prompts import SQL_SYSTEM, SQL_USER_TEMPLATE
from dataagent.generation.questions import schema_text
from dataagent.llm.base import ChatMessage, ModelClient, extract_json
from dataagent.logging_utils import get_logger
from dataagent.config import get_settings
from dataagent.schema.store import SchemaStore

log = get_logger(__name__)


class SchemaValidationError(Exception):
    pass


class SQLGenerationEngine:
    def __init__(self, model: ModelClient, store: SchemaStore) -> None:
        self._model = model
        self._store = store

    def generate(self, question: CandidateQuestion) -> SQLCandidate | None:
        datasets = self._store.list_datasets()
        by_fp = {d["fingerprint"]: d for d in datasets}
        wanted = [by_fp[fp] for fp in question.dataset_fingerprints if fp in by_fp]
        text = schema_text(wanted or datasets, self._store.list_relationships())
        user = SQL_USER_TEMPLATE.format(
            schema_text=text,
            question=question.question,
            category=question.category,
            difficulty=question.difficulty,
            expected_sql_pattern=question.expected_sql_pattern,
            max_rows=get_settings().query_max_rows,
        )
        resp = self._model.complete(
            [ChatMessage("system", SQL_SYSTEM), ChatMessage("user", user)],
            temperature=0.35, max_tokens=2048)
        try:
            payload = extract_json(resp.content)
        except ValueError as exc:
            log.warning("SQL output was not JSON: %s", exc)
            raise SchemaValidationError("model returned unparseable JSON") from exc
        try:
            jsonschema.validate(payload, SQL_OUTPUT_SCHEMA)
        except jsonschema.ValidationError as exc:
            log.warning("SQL JSON failed schema validation: %s", exc.message)
            raise SchemaValidationError(f"json_schema: {exc.message}") from exc
        return SQLCandidate(
            dataset_fingerprints=question.dataset_fingerprints,
            category=question.category,
            expected_sql_pattern=question.expected_sql_pattern,
            **payload,
        )


class TemplateSQLGenerator:
    """Writes deterministic Dremio SQL per question category (mock mode)."""

    def __init__(self, store: SchemaStore) -> None:
        self._store = store

    def generate(self, question: CandidateQuestion) -> SQLCandidate | None:
        datasets = self._store.list_datasets()
        by_fp = {d["fingerprint"]: d for d in datasets}
        wanted = [by_fp[fp] for fp in question.dataset_fingerprints if fp in by_fp]
        ds = wanted[0] if wanted else None
        if ds is None:
            return None
        cols = ds["columns"]
        col_names = [c["name"] for c in cols]
        qp = ".".join(ds["path"])
        tbl_cols = {ds["name"]: set(col_names)}

        def pick(pred, default=None):
            for c in col_names:
                if pred(c):
                    return c
            return default

        money = pick(lambda c: any(k in c for k in ("amount", "price", "cost", "revenue", "refund")))
        datec = pick(lambda c: "date" in c or "timestamp" in c)
        strc = pick(lambda c: "status" in c or "category" in c or "region" in c)
        idc = pick(lambda c: c.endswith("_id"), col_names[0])
        cat = question.category
        if cat in ("sorting_limiting", "aggregation", "window_functions", "ranking",
                   "business_metric") and money is None:
            return None
        if cat in ("date_timestamp", "period_comparison") and datec is None:
            return None
        if cat in ("filtering", "group_by_having", "null_handling") and strc is None:
            return None

        if cat == "basic_selection":
            c = pick(lambda n: any(k in n for k in ("name", "email", "category", "region")))
            sql = f'SELECT {c} FROM {qp} LIMIT 10'
        elif cat == "filtering":
            c = pick(lambda n: "status" in n or "tier" in n or "region" in n)
            sql = f"SELECT COUNT(*) AS n FROM {qp} WHERE {c} = 'X' LIMIT 1"
        elif cat == "sorting_limiting":
            sql = f"SELECT {idc}, {money} FROM {qp} ORDER BY {money} DESC LIMIT 5"
        elif cat == "aggregation":
            sql = (f"SELECT SUM({money}) AS s, AVG({money}) AS a, COUNT(*) AS n"
                   f" FROM {qp} LIMIT 1")
        elif cat == "group_by_having":
            sql = (f"SELECT {strc}, COUNT(*) AS n FROM {qp} GROUP BY {strc}"
                   f" HAVING COUNT(*) > 10 LIMIT 20")
        elif cat == "date_timestamp":
            sql = (f"SELECT DATE_TRUNC('MONTH', {datec}) AS month, COUNT(*) AS n"
                   f" FROM {qp} GROUP BY 1 ORDER BY 1 LIMIT 24")
        elif cat == "joins":
            rels = [r for r in self._store.list_relationships()
                    if ds["fingerprint"] in (r["left"], r["right"])]
            if not rels:
                # no verified relationship evidence -> join question cannot
                # be safely answered; skip (never infer from column names)
                return None
            rel = rels[0]
            left_col = rel["left_columns"][0]
            right_col = rel["right_columns"][0]
            other_fp = rel["right"] if rel["left"] == ds["fingerprint"] else rel["left"]
            other = by_fp[other_fp]
            sql = (f"SELECT a.{left_col}, b.{right_col}"
                   f" FROM {qp} a JOIN {'.'.join(other['path'])} b"
                   f" ON a.{left_col} = b.{right_col} LIMIT 10")
        elif cat == "cte":
            avg_src = money
            sql = (f"WITH avg_val AS (SELECT AVG({avg_src}) AS a FROM {qp})"
                   f" SELECT {qp}.{idc}, {qp}.{avg_src} FROM {qp}, avg_val"
                   f" WHERE {qp}.{avg_src} < avg_val.a LIMIT 10")
        elif cat == "window_functions":
            sql = (f"SELECT {idc}, SUM({money}) OVER (ORDER BY {idc}) AS running_total"
                   f" FROM {qp} ORDER BY {idc} LIMIT 10")
        elif cat == "null_handling":
            sql = f'SELECT COUNT(*) AS n FROM {qp} WHERE {strc} IS NULL LIMIT 1'
        elif cat == "ranking":
            sql = (f"SELECT {idc}, {money}, RANK() OVER (ORDER BY {money} DESC) AS r"
                   f" FROM {qp} LIMIT 3")
        elif cat == "period_comparison":
            sql = (f"SELECT DATE_TRUNC('MONTH', {datec}) AS month, COUNT(*) AS n"
                   f" FROM {qp} GROUP BY 1 ORDER BY 1 LIMIT 24")
        elif cat == "business_metric":
            sql = f"SELECT SUM({money}) AS total FROM {qp} LIMIT 1"
        elif cat in ("schema_ambiguity", "missing_info", "unsupported"):
            return None  # intentionally unanswerable: not exported as SQL
        else:
            return None

        # every generated SQL must parse; template bugs fail fast
        sqlglot.parse_one(sql, read="dremio")
        return SQLCandidate(
            question=question.question,
            required_tables=[qp],
            required_columns=sorted({c.name for c in sqlglot.parse_one(sql, read="dremio").find_all(exp.Column)}),
            sql=sql,
            task_type=question.task_type,
            difficulty=question.difficulty,
            category=question.category,
            expected_sql_pattern=question.expected_sql_pattern,
            dataset_fingerprints=question.dataset_fingerprints,
        )
