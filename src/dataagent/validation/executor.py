"""Stages 6-7: safe execution of validated candidate SQL.

Execution is opt-in (settings.enable_query_execution), runs only against the
configured test Dremio with a read-only account, and enforces timeouts and a
hard row limit. Every execution is recorded in the audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataagent.config import Settings, get_settings
from dataagent.dremio.client import (
    AuthenticationError,
    DremioClient,
    DremioError,
    PermissionError,
    ResourceUnavailableError,
)
from dataagent.dremio.models import JobResult
from dataagent.logging_utils import AuditLog, get_logger

log = get_logger(__name__)


@dataclass
class ExecutionOutcome:
    executed: bool = False
    status: str = "not_executed"  # completed | failed | timed_out | disabled | rejected
    error_message: str | None = None
    error_class: str = ""
    job_id: str = ""
    rows: list = field(default_factory=list)
    row_count: int = 0


@dataclass
class CorrectedQuery:
    sql: str
    original_sql: str
    error_class: str
    error_message: str


class ExecutionService:
    def __init__(self, client: DremioClient | None = None,
                 settings: Settings | None = None,
                 audit: AuditLog | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client or DremioClient(self._settings)
        self._audit = audit or AuditLog()

    def enforce_limits(self, sql: str) -> str:
        """Wrap queries lacking a LIMIT with the configured hard LIMIT."""
        import sqlglot
        from sqlglot import exp as E
        try:
            tree = sqlglot.parse_one(sql, read="dremio")
            if not tree.find(E.Limit):
                tree = tree.limit(self._settings.query_max_rows)
                return tree.sql(dialect="dremio")
        except Exception:
            pass
        return sql

    def execute(self, sql: str, example_id: str = "") -> ExecutionOutcome:
        if not self._settings.enable_query_execution:
            return ExecutionOutcome(status="disabled")
        wrapped = self.enforce_limits(sql)
        try:
            result: JobResult = self._client.execute_sql(
                wrapped, timeout_seconds=self._settings.query_timeout_seconds)
        except AuthenticationError as exc:
            self._audit.record("execution_error", example_id=example_id, sql=wrapped,
                               error_class="authentication", error=str(exc))
            return ExecutionOutcome(executed=True, status="failed",
                                    error_class="authentication", error_message=str(exc))
        except PermissionError as exc:
            self._audit.record("execution_error", example_id=example_id, sql=wrapped,
                               error_class="permission", error=str(exc))
            return ExecutionOutcome(executed=True, status="failed",
                                    error_class="permission", error_message=str(exc))
        except ResourceUnavailableError as exc:
            self._audit.record("execution_error", example_id=example_id, sql=wrapped,
                               error_class="resource_unavailable", error=str(exc))
            return ExecutionOutcome(executed=True, status="failed",
                                    error_class="resource_unavailable", error_message=str(exc))
        except DremioError as exc:
            self._audit.record("execution_error", example_id=example_id, sql=wrapped,
                               error_class="dremio", error=str(exc))
            return ExecutionOutcome(executed=True, status="failed",
                                    error_class="dremio", error_message=str(exc))

        rows = result.rows[: self._settings.query_max_rows]
        outcome = ExecutionOutcome(
            executed=True,
            status=("completed" if result.state == "COMPLETED" else
                    "timed_out" if result.state == "TIMED_OUT" else "failed"),
            error_message=result.error_message,
            error_class=("sql" if result.state == "FAILED" else result.state.lower()),
            job_id=result.job_id,
            rows=rows,
            row_count=result.row_count,
        )
        self._audit.record("execution", example_id=example_id, sql=wrapped,
                           status=outcome.status, job_id=result.job_id,
                           rows_returned=len(rows))
        return outcome
