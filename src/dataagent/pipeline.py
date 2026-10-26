"""End-to-end pipeline orchestration.

generate(): schema-aware candidates -> JSON-Schema validation -> parser
validation (stages 2-5) -> correction loop -> staged execution (stage 6-7,
opt-in) -> semantic verification -> QC dedup -> grouped split -> JSONL export
-> acceptance report.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import sqlglot
from sqlglot import exp as E

from dataagent.config import Settings, get_settings, OUTPUT_DIR, REPORTS_DIR
from dataagent.dremio.client import DremioClient
from dataagent.errors.correction import CorrectionEngine, classify_error, NON_SQL_CLASSES
from dataagent.eval.report import ReportWriter
from dataagent.export.jsonl import JSONLExporter, build_user_content
from dataagent.generation.models import SQLCandidate
from dataagent.generation.questions import (
    QuestionEngine,
    TemplateQuestionEngine,
    schema_text,
    question_fingerprint,
)
from dataagent.generation.sql_gen import (
    SQLGenerationEngine,
    TemplateSQLGenerator,
    SchemaValidationError,
)
from dataagent.generation.triage import RuleBasedTriage, TriageService, triage_all
from dataagent.llm.base import TrackedModelClient, UsageTracker
from dataagent.llm.glm import build_model_client
from dataagent.logging_utils import AuditLog, get_logger
from dataagent.quality.qc import split_grouped, sql_fingerprint, question_fingerprint
from dataagent.quality.library import ExampleLibrary
from dataagent.quality.semantic_dedup import SemanticDedup
from dataagent.schema.store import SchemaStore
from dataagent.validation.evidence import EvidenceService
from dataagent.validation.executor import ExecutionService
from dataagent.validation.parser import ParserValidator
from dataagent.validation.semantic_checks import run_static_checks
from dataagent.validation.verification import SemanticReview

log = get_logger(__name__)

_LIMIT_QUESTION_RE = re.compile(
    r"\b(top|first|highest|lowest|newest|oldest|largest|smallest|last|most)\s+\d+\b"
    r"|\b\d+\s+(top|first|highest|lowest|newest|oldest|largest|smallest|most)\b"
    r"|\blimit\s+\d+\b|\b\d+\s+rows?\b",
    re.IGNORECASE)


def question_requests_limit(question: str) -> bool:
    return bool(_LIMIT_QUESTION_RE.search(question))


def contains_sql_comment(sql: str) -> bool:
    return bool(re.search(r"(--[^\n])|(/\*)", sql))


def sql_limit_value(sql: str) -> int | None:
    try:
        tree = sqlglot.parse_one(sql, read="dremio")
    except Exception:
        return None
    limits = [s.args.get("limit") for s in tree.find_all(E.Select)]
    values = []
    for limit in limits:
        if limit is None:
            continue
        try:
            values.append(int(limit.expression.name))
        except (AttributeError, ValueError):
            return -1  # non-literal limit
    return max(values) if values else None


def strip_limit(sql: str) -> str:
    try:
        tree = sqlglot.parse_one(sql, read="dremio")
        for select in tree.find_all(E.Select):
            select.set("limit", None)
        return tree.sql(dialect="dremio")
    except Exception:
        return sql


class _StaticReject:
    """Stand-in for a ValidationVerdict with a fixed failure."""

    def __init__(self, reason: str) -> None:
        self.ok = False
        self.reason = reason


class DataGenerationPipeline:
    def __init__(self, settings: Settings | None = None,
                 library: ExampleLibrary | None = None) -> None:
        self._settings = settings or get_settings()
        self._store = SchemaStore()
        self._usage = UsageTracker()
        self._model = TrackedModelClient(build_model_client(self._settings), self._usage)
        self._mock = self._settings.glm_mode == "mock"
        self._audit = AuditLog()
        self._executor = ExecutionService(settings=self._settings, audit=self._audit)
        self._evidence = EvidenceService(self._executor)
        self._correction = CorrectionEngine(self._model, self._settings)
        self._library = library or ExampleLibrary()
        self._export_dir = OUTPUT_DIR
        self._reports_dir = REPORTS_DIR
        self._semantic = SemanticDedup(
            threshold=self._settings.semantic_similarity_threshold)
        self._sync_semantic_index()

    def _sync_semantic_index(self) -> None:
        """Ensure the embedding index covers all library questions (opt-in)."""
        if not (self._settings.semantic_dedup_enabled and self._semantic.available):
            return
        rows = self._library.all_examples()
        if self._semantic.count() >= len(rows):
            return
        for row in rows:
            self._semantic.add(row["question"], row["example_id"])
        log.info("Semantic index synced: %d questions", self._semantic.count())

    # -- one run ---------------------------------------------------------------

    def run(self, target: int, fresh: bool = False) -> dict:
        started = time.monotonic()
        datasets = self._store.list_datasets()
        if not datasets:
            raise RuntimeError(
                "No datasets cached; run `dataagent discover` first.")
        if fresh:
            self._library.reset()
        before = self._library.counts()["total"]

        questions, engine_q, question_failures = self._questions(target, datasets)

        # Triage: only `ready` questions proceed to SQL generation.
        triage_block = schema_text(datasets, self._store.list_relationships())
        triage = RuleBasedTriage() if self._mock else TriageService(self._model)
        triage_all(triage, questions, triage_block, self._settings.pipeline_workers)
        ready = [q for q in questions if q.triage_verdict == "ready"]
        ambiguous = [q for q in questions if q.triage_verdict == "needs_clarification"]
        unsupported = [q for q in questions if q.triage_verdict == "unsupported"]
        self._audit.record("triage", ready=len(ready),
                           needs_clarification=len(ambiguous),
                           unsupported=len(unsupported))

        candidates = self._sql_and_validate(ready, datasets)
        accepted, rejected, clarifications = self._verify_qc(candidates, ambiguous)
        split_counts, outputs = self._export(accepted, clarifications)

        # Persist verified examples cumulatively; rebuild splits from library.
        added = self._library.add_examples(accepted)
        cumulative = self._library.rebuild_splits(
            out_dir=self._export_dir, version=self._settings.dataset_version)

        report = ReportWriter(reports_dir=self._reports_dir,
                              version=self._settings.dataset_version)
        usage = self._usage.snapshot()
        pricing = (self._settings.llm_input_price_per_m,
                   self._settings.llm_output_price_per_m)
        summary = ReportWriter.summarize(
            [q.question for q in questions], accepted, dict(rejected), split_counts)
        summary["question_generation_failures"] = question_failures
        summary["tokens"] = usage
        summary["estimated_cost_usd"] = round(self._usage.estimated_cost(*pricing), 4)
        summary["estimated_cost_per_accepted_usd"] = round(
            self._usage.estimated_cost(*pricing) / max(1, len(accepted)), 4)
        summary["model"] = {"generator": self._model.model_id,
                            "reviewer": self._model.model_id}
        summary["cumulative"] = {"before_this_run": before,
                                 "added_this_run": added,
                                 "total_verified": cumulative.get("train", 0)
                                 + cumulative.get("validation", 0)
                                 + cumulative.get("test", 0),
                                 "by_split": cumulative}
        path = report.write(summary)
        outputs["report"] = str(path)
        outputs["elapsed_seconds"] = round(time.monotonic() - started, 1)
        outputs["workers"] = self._settings.pipeline_workers
        outputs["added_this_run"] = added
        outputs["cumulative_total"] = summary["cumulative"]["total_verified"]
        log.info("Pipeline complete: %d candidates, %d accepted (%.1fs)",
                 len(questions), len(accepted), outputs["elapsed_seconds"])
        return outputs | {"split_counts": split_counts}

    def _questions(self, target: int, datasets):
        if self._mock:
            engine_q = TemplateQuestionEngine(self._store)
        else:
            engine_q = QuestionEngine(self._model, self._store)
        # diversity context: questions already in the library are shown to the
        # generator so it avoids similar wording/metrics/filters
        existing = [row["question"] for row in self._library.all_examples()][-25:]
        batch = self._settings.generation_batch_size
        batches = []
        for start in range(0, target, batch):
            batches.append(batch)
        workers = max(1, self._settings.pipeline_workers)
        all_q: list = []
        failures: list[str] = []

        def gen(n):
            try:
                return engine_q.generate(n, datasets=datasets,
                                         existing_questions=existing), None
            except SchemaValidationError as exc:
                return [], f"invalid_question_json: {exc}"

        if len(batches) <= 1 or self._mock:
            for b in batches:
                qs, err = gen(b)
                all_q.extend(qs)
                if err:
                    failures.append(err)
        else:
            # parallel batch generation; dedup happens afterwards
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for qs, err in pool.map(gen, batches):
                    all_q.extend(qs)
                    if err:
                        failures.append(err)
        return all_q, engine_q, failures
    def _sql_and_validate(self, questions, datasets):
        sql_engine = (TemplateSQLGenerator(self._store) if self._mock
                      else SQLGenerationEngine(self._model, self._store))
        parser = ParserValidator(max_rows=self._settings.query_max_rows)

        known_tables = {".".join(d["path"]).lower() for d in datasets}
        known_tables |= {d["name"].lower() for d in datasets}  # unqualified names
        known_columns = {d["name"].lower(): {c["name"].lower() for c in d["columns"]}
                         for d in datasets}

        def process(q):
            try:
                cand = sql_engine.generate(q)
            except SchemaValidationError as exc:
                return {"question": q.question, "category": q.category,
                        "difficulty": q.difficulty, "task_type": q.task_type,
                        "candidate": None, "dataset_fingerprints": q.dataset_fingerprints,
                        "parse_verdict": _StaticReject(str(exc)),
                        "validation_status": "invalid_json"}
            if cand is None:
                return None
            cand = self._clamp_limit(cand)
            verdict = self._validate_candidate(parser, cand, known_tables, known_columns)
            return {
                "question": q.question,
                "sql": cand.sql,
                "candidate": cand,
                "category": q.category,
                "difficulty": q.difficulty,
                "task_type": q.task_type,
                "dataset_fingerprints": q.dataset_fingerprints,
                "parse_verdict": verdict,
                "validation_status": "executing" if verdict.ok else "rejected_parse",
            }

        workers = max(1, self._settings.pipeline_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            processed = list(pool.map(process, questions))
        return [c for c in processed if c is not None]

    def _clamp_limit(self, cand: SQLCandidate) -> SQLCandidate:
        """Clamp over-large LIMITs to the configured max rows."""
        import sqlglot
        from sqlglot import exp as E

        try:
            tree = sqlglot.parse_one(cand.sql, read="dremio")
        except Exception:
            return cand
        changed = False
        for select in tree.find_all(E.Select):
            limit = select.args.get("limit")
            if limit is None:
                continue
            try:
                rows = int(limit.expression.name)
            except (AttributeError, ValueError):
                continue
            if rows > self._settings.query_max_rows:
                select.set("limit", E.Limit(expression=E.Literal.number(self._settings.query_max_rows)))
                changed = True
        if changed:
            cand = cand.model_copy(update={"sql": tree.sql(dialect="dremio")})
        return cand

    def try_correct(self, item: dict, error_message: str, schema_block: str):
        """Correction hook: returns a corrected candidate dict or None."""
        cand: SQLCandidate = item["candidate"]
        if self._mock:
            return None
        corrected = self._correction.correct(
            question=item["question"], sql=cand.sql,
            error_message=error_message, schema_text=schema_block)
        if not corrected:
            return None
        new_cand = cand.model_copy(update={"sql": corrected})
        return new_cand

    def execute_and_verify(self, item: dict, datasets):
        """Stages 6-7 plus completeness, semantic checks & review. Returns dict."""
        cand: SQLCandidate = item["candidate"]
        category = item["category"]
        question = item["question"]

        # Completeness check: LIMIT must be requested by the question.
        requests_limit = question_requests_limit(question)
        sql_limit = sql_limit_value(cand.sql)
        if sql_limit is not None and not requests_limit:
            # Question asks for all rows: strip the arbitrary LIMIT from the
            # exported SQL (execution still applies the hard row cap).
            item["candidate"] = cand.model_copy(
                update={"sql": strip_limit(cand.sql)})
        elif sql_limit is None and requests_limit:
            item["rejection_reason"] = "missing_requested_limit"
            return None  # question asks for a top-N result; SQL doesn't limit it

        # Training-cleanliness: no SQL comments in the assistant target.
        if contains_sql_comment(item["candidate"].sql):
            item["rejection_reason"] = "sql_contains_comments"
            return None

        # Deterministic semantic checks (pre-execution).
        uniqueness = self._join_uniqueness(item, datasets)
        findings = run_static_checks(question, item["candidate"].sql, datasets,
                                     uniqueness, evidence=self._evidence
                                     if hasattr(self, "_evidence") else None)
        static_needs_review = [f["reason"] for f in findings
                               if f["severity"] == "needs_review"]

        outcome = self._executor.execute(item["candidate"].sql, example_id=question[:40])
        if outcome.status == "failed" and outcome.error_class in NON_SQL_CLASSES | {"unknown"}:
            item["validation_status"] = "failed_execution"
            return None
        if outcome.status == "failed" and not self._mock:
            # correction loop (bounded)
            schema_block = self._schema_block_for(item, datasets)
            new_cand = self.try_correct(item, outcome.error_message or "", schema_block)
            if new_cand is not None:
                item["candidate"] = new_cand
                item["original_sql"] = cand.sql
                item["correction_error"] = outcome.error_message
                item["original_error"] = outcome.error_message
                item["original_error_class"] = outcome.error_class
                outcome = self._executor.execute(new_cand.sql)
        semantic = self._verify_semantics(item, datasets, outcome.rows)
        if semantic:
            semantic["reasons"] = list(semantic.get("reasons", []) or []) + static_needs_review
            if static_needs_review and semantic.get("verdict") == "accept":
                semantic["verdict"] = "needs_review"

        validation_status = "passed" if outcome.status in ("completed", "disabled", "not_executed") else outcome.status
        example = {
            "question": question,
            "category": category,
            "difficulty": item["difficulty"],
            "task_type": item["task_type"],
            "dataset_fingerprints": item["dataset_fingerprints"],
            "assistant_sql": item["candidate"].sql,
            "candidate_sql": item["candidate"].sql,
            "validation_status": validation_status,
            "execution_status": outcome.status,
            "semantic_status": semantic.get("verdict") if semantic else "skipped",
            "semantic_reasons": (semantic or {}).get("reasons", []),
            "reference_comparison": (semantic or {}).get("reference_comparison", "not_applicable"),
            "semantic_findings": static_needs_review,
            "generation_model": self._model.model_id,
        }
        return example

    def reject_statically(self, item: dict, reason: str):
        return {"question": item["question"], "category": item["category"],
                "difficulty": item["difficulty"], "task_type": item["task_type"],
                "assistant_sql": "", "validation_status": f"rejected: {reason}",
                "execution_status": "not_executed", "semantic_status": "skipped",
                "generation_model": self._model.model_id}

    def _validate_candidate(self, parser, cand, known_tables, known_columns):
        verdict = parser.parse_and_validate(cand.sql, known_tables=known_tables,
                                            known_columns=known_columns)
        return verdict

    def _verify_semantics(self, item, datasets, rows):
        if self._mock:
            return {"verdict": "accept" if rows is not None else "needs_review",
                    "reasons": ["rule-based check (mock mode)"],
                    "reference_comparison": "not_applicable"}
        reviewer = SemanticReview(self._model, self._executor, self._audit)
        schema_block = self._schema_block_for(item, datasets)
        return reviewer.verify(question=item["question"], sql=item["candidate"].sql,
                               category=item["category"], task_type=item["task_type"],
                               schema_block=schema_block, result_rows=rows,
                               datasets=datasets, evidence=self._evidence)

    def _join_uniqueness(self, item, datasets) -> dict[str, bool]:
        """Executed uniqueness evidence for join keys used by this candidate."""
        import sqlglot
        from sqlglot import exp as E

        try:
            tree = sqlglot.parse_one(item["candidate"].sql, read="dremio")
        except Exception:
            return {}
        if not tree.find(E.Join) or not getattr(self, "_evidence", None):
            return {}
        uniqueness: dict[str, bool] = {}
        fp_by_alias: dict[str, str] = {}
        for tbl in tree.find_all(E.Table):
            fp = next((d["fingerprint"] for d in datasets
                       if d["name"].lower() == tbl.name.lower()), None)
            if fp:
                fp_by_alias[tbl.alias_or_name.lower()] = fp
                fp_by_alias[tbl.name.lower()] = fp
        for join in tree.find_all(E.Join):
            cond = join.args.get("on")
            if not (isinstance(cond, E.EQ) and isinstance(cond.this, E.Column)
                    and isinstance(cond.expression, E.Column)):
                continue
            for col in (cond.this, cond.expression):
                fp = fp_by_alias.get((col.table or "").lower())
                if not fp:
                    continue
                ds = next(d for d in datasets if d["fingerprint"] == fp)
                unique = self._evidence.column_uniqueness(
                    ds["fingerprint"], ".".join(ds["path"]), col.name)
                if unique is not None:
                    uniqueness[f"{ds['fingerprint']}.{col.name.lower()}"] = unique
        return uniqueness

    def _schema_block_for(self, item, datasets):
        fingerprints = set(item.get("dataset_fingerprints") or [])
        wanted = [d for d in datasets if d["fingerprint"] in fingerprints]
        if not wanted:
            # derive from the SQL's parsed table references, never from guesses
            wanted = self._tables_from_sql(item["candidate"].sql, datasets)
        if not wanted:
            wanted = [d for d in datasets if d["fingerprint"].startswith("ecommerce")]
        all_rels = self._store.list_relationships()
        wanted_fps = {d["fingerprint"] for d in wanted}
        relevant = [r for r in all_rels
                    if r["left"] in wanted_fps or r["right"] in wanted_fps]
        return schema_text(wanted, relevant)

    def _tables_from_sql(self, sql: str, datasets):
        import sqlglot
        from sqlglot import exp as E

        try:
            tree = sqlglot.parse_one(sql, read="dremio")
        except Exception:
            return []
        cte_names = {c.alias_or_name.lower() for c in tree.find_all(E.CTE)}
        by_name = {d["name"].lower(): d for d in datasets}
        wanted = []
        for tbl in tree.find_all(E.Table):
            name = tbl.name.lower()
            if name in cte_names:
                continue
            ds = by_name.get(name)
            if ds and ds not in wanted:
                wanted.append(ds)
        return wanted

    # -- QC + export ------------------------------------------------------------

    def _verify_qc(self, candidates, ambiguous_questions=None):
        """Parallel execution + semantic verification, then ordered dedup QC.

        Dedup is cumulative: fingerprints already in the ExampleLibrary (from
        previous runs) reject new candidates as duplicate_across_runs.
        """
        datasets = self._store.list_datasets()
        accepted = []
        rejected: dict[str, list[str]] = defaultdict(list)
        clarifications: list[dict] = []
        lib_sql = self._library.seen_sql_fingerprints()
        lib_questions = self._library.seen_question_fingerprints()

        def run_item(item):
            if not item["parse_verdict"].ok:
                return None
            return self.execute_and_verify(item, datasets)

        workers = max(1, self._settings.pipeline_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            examples = list(pool.map(run_item, candidates))

        seen_sql: set[str] = set()
        seen_questions: set[str] = set()
        example_id = 0
        for item, example in zip(candidates, examples):
            example_id += 1
            if not item["parse_verdict"].ok:
                reason = item["parse_verdict"].reason or "parse"
                rejected[reason].append(item["question"])
                continue
            if example is None:
                rejected[item.get("rejection_reason", "failed_execution")].append(item["question"])
                continue
            if example["execution_status"] in ("failed", "timed_out"):
                rejected["failed_execution"].append(item["question"])
                self._audit.record("rejected_example", question=example["question"],
                                   reason="failed_execution", sql=example["assistant_sql"])
                continue
            verdict = example["semantic_status"]
            fp = sql_fingerprint(example["assistant_sql"])
            qfp = question_fingerprint(example["question"])
            if verdict == "reject":
                reasons = example.get("semantic_reasons", [])
                if any("reference check" in r for r in reasons):
                    rejected["reference_mismatch"].append(example["question"])
                else:
                    rejected["semantic_reject"].append(example["question"])
                self._audit.record("rejected_example", question=example["question"],
                                   reason="semantic_reject",
                                   reasons=reasons,
                                   sql=example["assistant_sql"])
                continue
            if fp in seen_sql:
                rejected["duplicate_sql"].append(example["question"])
                continue
            if qfp in seen_questions:
                rejected["duplicate_question"].append(example["question"])
                continue
            if fp in lib_sql or qfp in lib_questions:
                rejected["duplicate_across_runs"].append(example["question"])
                self._audit.record("rejected_example", question=example["question"],
                                   reason="duplicate_across_runs")
                continue
            # layer 3 (opt-in): embedding-based paraphrase detection
            if self._settings.semantic_dedup_enabled and self._semantic.available:
                match = self._semantic.check(example["question"])
                if match.similar:
                    rejected["near_duplicate_semantic"].append(
                        example["question"])
                    self._audit.record(
                        "rejected_example", question=example["question"],
                        reason="near_duplicate_semantic",
                        similarity=round(match.score, 3),
                        matched=match.matched_example_id)
                    continue
            seen_sql.add(fp)
            seen_questions.add(qfp)
            schema_group = ",".join(sorted(item["dataset_fingerprints"])) or "unknown"
            example["example_id"] = f"ex_{example_id:06d}"
            example["sql_fingerprint"] = fp
            example["schema_group"] = schema_group
            example["user_content"] = build_user_content(
                self._schema_block_for(item, datasets), example["question"])
            if verdict == "needs_review":
                example["needs_review"] = True
            accepted.append(example)
            self._audit.record("accepted_example", example_id=example["example_id"],
                               sql=example["assistant_sql"], category=example["category"])
            self._semantic.add(example["question"], example["example_id"])

        # clarification examples for triaged-ambiguous questions
        for q in ambiguous_questions or []:
            clarifications.append({
                "question": q.question,
                "category": q.category,
                "difficulty": q.difficulty,
                "task_type": q.task_type,
                "interpretations": q.triage_interpretations,
                "triage_reason": q.triage_reason,
                "generation_model": self._model.model_id,
            })
            rejected["needs_clarification"].append(q.question)
        return accepted, rejected, clarifications

    def _export(self, accepted, clarifications=None):
        promoted = [e for e in accepted if not e.get("needs_review")]
        review = [e for e in accepted if e.get("needs_review")]
        splits = split_grouped(promoted)
        exporter = JSONLExporter(out_dir=self._export_dir,
                                 version=self._settings.dataset_version)
        outputs = {}
        for split in ("train", "validation", "test"):
            sft, man = exporter.write_split(split, splits[split])
            outputs[f"{split}_sft"] = str(sft)
            outputs[f"{split}_manifest"] = str(man)
        if review:
            nrv, nrm = exporter.write_split("needs_review", review)
            outputs["needs_review_sft"] = str(nrv)
            outputs["needs_review_manifest"] = str(nrm)
        if clarifications and self._settings.export_clarifications:
            clv, clm = exporter.write_split("clarifications", clarifications)
            outputs["clarifications_sft"] = str(clv)
            outputs["clarifications_manifest"] = str(clm)
        counts = {k: len(v) for k, v in splits.items()}
        counts["needs_review"] = len(review)
        counts["clarifications"] = len(clarifications or [])
        return counts, outputs
