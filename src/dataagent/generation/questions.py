"""Question-generation engine.

LLM path uses GLM with the schema block. The template path produces
deterministic questions without any model call — used in mock/dev mode and by
tests, and guarantees category coverage.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable

from dataagent.generation.models import SQL_CATEGORIES, DIFFICULTIES, CandidateQuestion
from dataagent.llm.base import ChatMessage, ModelClient, extract_json
from dataagent.generation.prompts import QUESTION_SYSTEM, QUESTION_USER_TEMPLATE
from dataagent.logging_utils import get_logger
from dataagent.schema.store import SchemaStore

log = get_logger(__name__)

STOP = set("a an the of from in on to with for and or by is are was were what which how many "
           "much show list give get all".split())
_token_re = re.compile(r"[a-z0-9]+")


def question_fingerprint(text: str) -> str:
    """Lexical near-duplicate fingerprint (sorted content tokens)."""
    tokens = sorted(set(_token_re.findall(text.lower())) - STOP)
    return "|".join(tokens[:20])


def schema_text(datasets: list[dict], relationships: list[dict] | None = None,
                max_datasets: int = 6) -> str:
    """Render schema metadata for prompts (documented evidence only).

    Includes only datasets passed in; relationships are deduplicated and
    restricted to pairs where BOTH sides are among the included datasets.
    """
    blocks = []
    for ds in datasets[:max_datasets]:
        cols = "\n".join(f"  {c['name']}: {c['type']}"
                         + (f"  -- {c['description']}" if c.get("description") else "")
                         for c in ds["columns"])
        desc = f"\nDescription: {ds['description']}" if ds.get("description") else ""
        blocks.append(f'Dataset: {".".join(ds["path"])} ({ds["dataset_type"]}){desc}\n{cols}')
    text = "\n\n".join(blocks)
    if relationships:
        included = {d["fingerprint"] for d in datasets[:max_datasets]}
        seen: set[tuple] = set()
        rendered = []
        for r in relationships:
            pair = (r["left"], tuple(r["left_columns"]), r["right"], tuple(r["right_columns"]))
            pair = tuple(sorted(pair, key=str))
            if pair in seen:
                continue
            if included and r["left"] not in included and r["right"] not in included:
                continue
            seen.add(pair)
            rendered.append(
                f'  {r["left"]}.{",".join(r["left_columns"])} -> {r["right"]}.{",".join(r["right_columns"])}'
                f' (source: {r["source"]})')
        if rendered:
            text += "\n\nDocumented relationships (only joins relevant to the above datasets):\n" \
                    + "\n".join(rendered)
    return text


class QuestionEngine:
    def __init__(self, model: ModelClient, store: SchemaStore) -> None:
        self._model = model
        self._store = store
        self._seen: set[str] = set()

    def generate(self, count: int, datasets: list[dict] | None = None,
                 categories: list[str] | None = None,
                 existing_questions: list[str] | None = None) -> list[CandidateQuestion]:
        datasets = datasets or self._store.list_datasets()
        categories = categories or SQL_CATEGORIES
        text = schema_text(datasets, self._store.list_relationships())
        cat_cycle = ", ".join(categories[:8])
        shown = (existing_questions or [])[:25]
        existing_block = "\n".join(f"- {q}" for q in shown) or "- (none yet)"
        user = QUESTION_USER_TEMPLATE.format(
            count=count, schema_text=text, categories=cat_cycle,
            existing_questions=existing_block)
        resp = self._model.complete(
            [ChatMessage("system", QUESTION_SYSTEM), ChatMessage("user", user)],
            temperature=0.85, max_tokens=6144)
        try:
            payload = extract_json(resp.content)
        except ValueError as exc:
            log.warning("Question output was not JSON: %s", exc)
            return []
        return self._validate_batch(payload, datasets)

    def _validate_batch(self, payload, datasets) -> list[CandidateQuestion]:
        path_to_fp = {d["fingerprint"]: d["fingerprint"] for d in datasets}
        known_paths = {d["name"]: ".".join(d["path"]) for d in datasets}
        out = []
        if not isinstance(payload, list):
            return out
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "question" not in item or "category" not in item:
                continue
            fp = question_fingerprint(item["question"])
            if fp in self._seen:
                continue
            self._seen.add(fp)
            tables = [t for t in item.get("required_tables", []) if t in known_paths.values()
                      or t in known_paths]
            item["required_tables"] = tables
            # map table paths -> cached dataset fingerprints
            fingerprints = []
            for t in tables:
                t_path = known_paths.get(t, t)
                if t_path.lower() in path_to_fp:
                    fingerprints.append(t_path.lower())
            item["task_type"] = str(item.get("task_type") or item.get("category") or "unknown")
            q = CandidateQuestion(
                dataset_fingerprints=fingerprints,
                **{k: item[k] for k in
                   ("question", "category", "difficulty", "task_type",
                    "required_tables", "required_columns", "expected_sql_pattern")
                   if k in item},
            )
            out.append(q)
        return out


class TemplateQuestionEngine:
    """Deterministic question factory (mock mode: no model, no network)."""

    LEVEL_MIX = ["beginner", "intermediate", "advanced"]

    def __init__(self, store: SchemaStore) -> None:
        self._store = store
        self._seen: set[str] = set()

    def generate(self, count: int, datasets: list[dict] | None = None,
                 categories: list[str] | None = None,
                 existing_questions: list[str] | None = None) -> list[CandidateQuestion]:
        datasets = datasets or self._store.list_datasets()
        categories = categories or SQL_CATEGORIES
        if not datasets:
            return []
        out: list[CandidateQuestion] = []
        i = 0
        last_out_len = 0
        since_progress = 0
        while len(out) < count and since_progress <= 2 * len(datasets):
            ds = datasets[i % len(datasets)]
            category = categories[(len(out)) % len(categories)]
            difficulty = self.LEVEL_MIX[len(out) % 3]
            q = self._template(ds, datasets, category, difficulty, i)
            if q is None:
                i += 1
                continue
            fp = question_fingerprint(q.question)
            if fp in self._seen:
                i += 1
                since_progress += 1
                continue
            self._seen.add(fp)
            out.append(q)
            i += 1
            since_progress = 0
            last_out_len = len(out)
        return out

    # -- per-category templates ------------------------------------------------

    def _template(self, ds: dict, all_datasets: list[dict], category: str,
                  difficulty: str, idx: int) -> CandidateQuestion | None:
        cols = {c["name"]: c["type"].upper() for c in ds["columns"]}
        qp = ".".join(ds["path"])
        dsn = ds["name"]

        def mk(question, task, tables, columns, pattern, partner=None):
            fps = {ds["fingerprint"]}
            if partner is not None:
                fps.add(partner["fingerprint"])
            return CandidateQuestion(
                question=question, category=category, difficulty=difficulty,
                task_type=task, required_tables=tables,
                required_columns=columns, expected_sql_pattern=pattern,
                dataset_fingerprints=sorted(fps),
            )

        def col(pred: Callable[[str], bool]) -> str | None:
            for name in cols:
                if pred(name):
                    return name
            return None

        if category == "basic_selection":
            c = col(lambda n: any(k in n for k in ("name", "email", "category", "region")))
            if not c:
                return None
            return mk(f"List {c} values from {dsn} for the first 10 records.",
                      "selection", [qp], [c], f'SELECT {c} FROM {qp} LIMIT 10')
        if category == "filtering":
            c = col(lambda n: "status" in n or "tier" in n or "region" in n)
            if not c:
                return None
            return mk(f"How many rows in {dsn} have {c} equal to a specific value, say X?",
                      "filtering", [qp], [c], f'SELECT COUNT(*) FROM {qp} WHERE {c} = X')
        if category == "sorting_limiting":
            c = col(lambda n: "amount" in n or "price" in n or "cost" in n)
            if not c:
                return None
            return mk(f"Show the 5 rows of {dsn} with the highest {c}.",
                      "top_n", [qp], [c], f"SELECT * FROM {qp} ORDER BY {c} DESC LIMIT 5")
        if category == "aggregation":
            c = col(lambda n: "amount" in n or "price" in n or "cost" in n)
            if not c:
                return None
            return mk(f"What are the SUM, AVG and COUNT of {c} in {dsn}?",
                      "aggregation", [qp], [c], f'SELECT SUM({c}), AVG({c}), COUNT(*) FROM {qp}')
        if category == "group_by_having":
            c = col(lambda n: "status" in n or "category" in n or "region" in n)
            if not c:
                return None
            return mk(f"Count rows per {c} in {dsn}, but only groups with more than 10 rows.",
                      "group_having", [qp], [c], f"SELECT {c}, COUNT(*) FROM {qp} GROUP BY {c} HAVING COUNT(*) > 10")
        if category == "date_timestamp":
            c = col(lambda n: "date" in n or "timestamp" in n)
            if not c:
                return None
            return mk(f"How many rows of {dsn} were created per month based on {c}?",
                      "date_analysis", [qp], [c], f"SELECT DATE_TRUNC('MONTH', {c}), COUNT(*) FROM {qp} GROUP BY 1")
        if category == "joins":
            partner = next((d for d in all_datasets if d is not ds and d.get("fingerprint")), None)
            if not partner:
                return None
            return mk(f"Join {dsn} with {partner['name']} and list one shared field pair.",
                      "join", [qp, ".".join(partner["path"])], [],
                      f"SELECT ... FROM {qp} JOIN {partner['path'][-1]} ON <verified relationship>",
                      partner=partner)
        if category == "cte":
            c = col(lambda n: cols[n] in ("BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL"))
            if not c:
                return None
            return mk(f"Using a CTE, compute the average {c} of {dsn}, then list rows below that average.",
                      "cte", [qp], [c], f"WITH avg AS (SELECT AVG({c}) AS a FROM {qp}) SELECT * FROM {qp}, avg WHERE {c} < a")
        if category == "window_functions":
            c = col(lambda n: "date" in n or "amount" in n or "price" in n)
            if not c:
                return None
            return mk(f"Compute the running total of {c} over {dsn} ordered by id.",
                      "window", [qp], [c], f"SELECT SUM({c}) OVER (ORDER BY <id>) FROM {qp}")
        if category == "null_handling":
            c = col(lambda n: cols[n] in ("VARCHAR",))
            if not c:
                return None
            return mk(f"How many rows in {dsn} have a missing {c}?",
                      "null_check", [qp], [c], f"SELECT COUNT(*) FROM {qp} WHERE {c} IS NULL")
        if category == "ranking":
            c = col(lambda n: "amount" in n or "price" in n or "cost" in n)
            if not c:
                return None
            return mk(f"Rank rows of {dsn} by {c} and show the top 3 ranks.",
                      "ranking", [qp], [c], f"SELECT RANK() OVER (ORDER BY {c} DESC) r, {c} FROM {qp}")
        if category == "period_comparison":
            c = col(lambda n: "date" in n or "timestamp" in n)
            if not c:
                return None
            return mk(f"Compare row counts of {dsn} between the earliest and latest month based on {c}.",
                      "period_compare", [qp], [c],
                      f"SELECT month, COUNT(*) FROM {qp} GROUP BY month ORDER BY month")
        if category == "business_metric":
            c = col(lambda n: "amount" in n or "price" in n or "cost" in n)
            if not c:
                return None
            return mk(f"What is the total {c} across {dsn}? (business revenue-style metric)",
                      "metric", [qp], [c], f"SELECT SUM({c}) FROM {qp}")
        if category == "schema_ambiguity":
            return mk("Show me the most popular items with their totals.",
                      "ambiguous", [qp], [], "<ambiguous wording: which table/column>")
        if category == "missing_info":
            return mk("What was our revenue in the last period?",
                      "underspecified", [qp], [], "<missing: which column, which time bucket>")
        if category == "unsupported":
            return mk("Predict next month's revenue.",
                      "unsupported", [qp], [], "<forecasting not supported by schema-only SQL>")
        return None


def new_example_id() -> str:
    return "ex_" + uuid.uuid4().hex[:16]
