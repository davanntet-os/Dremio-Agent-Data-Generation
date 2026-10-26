"""Generation models and the output JSON Schema for SQL generation."""

from __future__ import annotations

from pydantic import BaseModel, Field

SQL_CATEGORIES = [
    "basic_selection",
    "filtering",
    "sorting_limiting",
    "aggregation",
    "group_by_having",
    "date_timestamp",
    "joins",
    "cte",
    "window_functions",
    "null_handling",
    "ranking",
    "period_comparison",
    "business_metric",
    "schema_ambiguity",
    "missing_info",
    "unsupported",
]

DIFFICULTIES = ["beginner", "intermediate", "advanced"]


class CandidateQuestion(BaseModel):
    question: str
    category: str = ""
    difficulty: str = "beginner"
    task_type: str = ""
    required_tables: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    expected_sql_pattern: str = ""
    dataset_fingerprints: list[str] = Field(default_factory=list)
    triage_verdict: str = ""
    triage_reason: str = ""
    triage_interpretations: list[str] = Field(default_factory=list)


class SQLCandidate(BaseModel):
    question: str
    required_tables: list[str]
    required_columns: list[str]
    sql: str
    task_type: str
    difficulty: str

    # generation metadata (never exported into the SFT assistant target)
    category: str = ""
    expected_sql_pattern: str = ""
    dataset_fingerprints: list[str] = Field(default_factory=list)


SQL_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["question", "required_tables", "required_columns", "sql",
                 "task_type", "difficulty"],
    "properties": {
        "question": {"type": "string", "minLength": 10},
        "required_tables": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "required_columns": {"type": "array", "items": {"type": "string"}},
        "sql": {"type": "string", "minLength": 10},
        "task_type": {"type": "string", "minLength": 3},
        "difficulty": {"enum": DIFFICULTIES},
    },
    "additionalProperties": False,
}
