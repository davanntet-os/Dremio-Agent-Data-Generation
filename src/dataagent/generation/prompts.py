"""Prompt templates. Only approved schema metadata is embedded — no data values.

System prompts enforce:
- Dremio SQL dialect only (SELECT / WITH, no DML/DDL).
- Fully quoted dataset references as given in the schema block.
- Structured JSON output (validated downstream with jsonschema).
"""

QUESTION_SYSTEM = """\
You generate diverse, realistic natural-language analytical questions that an \
analyst would ask against Dremio. You are given ONLY schema metadata \
(table names, columns, types, descriptions). Never reference data values. \
Return ONLY a JSON array — no prose, no markdown fences.\
"""

QUESTION_USER_TEMPLATE = """\
Analyze this Dremio schema and produce {count} distinct analytical questions.

Schema:
{schema_text}

Requirements:
- Cover at least these categories spread across levels: {categories}
- difficulty must be one of: beginner, intermediate, advanced
- "required_tables": fully quoted dataset paths exactly as in the schema block
- "required_columns": column names used by the question
- "expected_sql_pattern": short pattern of the ideal SQL, e.g. \
'SELECT SUM(x) FROM t WHERE cond'
- Vary phrasing; avoid near-duplicates or template repetition
- Special categories:
  * schema_ambiguity: questions whose wording makes table/column choice unclear
  * missing_info: questions missing a filter/detail needed for a precise answer
  * unsupported: questions asking for something impossible with this schema

Return a JSON array of objects with keys: question, category, difficulty, \
task_type, required_tables, required_columns, expected_sql_pattern.

IMPORTANT — questions we ALREADY have (do NOT generate anything similar \
in wording, metric, grouping or filters):
{existing_questions}\
"""

SQL_SYSTEM = """\
You write Dremio SQL. Use ONLY Dremio-supported syntax, only the provided \
schema, and fully quoted dataset paths exactly as they appear. Allowed \
statements: SELECT and WITH ... SELECT. Never write INSERT/UPDATE/DELETE/DDL. \
LIMIT rules: only include LIMIT when the question explicitly asks for a \
limited result (e.g. "top 5", "first 10 rows"); a single-row aggregate \
(SUM/AVG/COUNT without GROUP BY) never needs LIMIT; do not add LIMIT 100 \
arbitrarily when the question asks for all rows. \
Return ONLY a single JSON object — no prose, no markdown fences.\
"""

SQL_USER_TEMPLATE = """\
Schema (ONLY this metadata is available):

{schema_text}

Generate the Dremio SQL for this analytical question and return a JSON object \
with EXACTLY these keys:

{{
  "question": <the question, verbatim>,
  "required_tables": [<fully quoted dataset paths used>],
  "required_columns": [<column names used>],
  "sql": <the Dremio SQL query>,
  "task_type": <short task label>,
  "difficulty": <beginner|intermediate|advanced>
}}

Question: {question}
Question category: {category}
Difficulty level: {difficulty}
Intended SQL pattern: {expected_sql_pattern}
Constraint: use only listed tables/columns; do NOT invent schema objects. \
If the question asks for all rows, do NOT add a LIMIT clause.\
"""

REVIEW_SYSTEM = """\
You are a strict SQL correctness reviewer. You compare a natural-language \
question, a candidate Dremio SQL query, and the schema metadata. Decide \
whether the SQL is a faithful answer to the question. Return ONLY JSON.\
"""

REVIEW_USER_TEMPLATE = """\
Schema:
{schema_text}

Question: {question}
Candidate SQL:
{sql}
Question category: {category}
Task type: {task_type}

Check strictly:
(1) all tables/columns exist in the schema;
(2) the query answers the question exactly;
(3) filters match the question requirements;
(4) aggregations are semantically appropriate (e.g. growth asked as a \
percentage must be computed as a percentage, not an absolute change; averages \
must be over the population the question implies — ordering customers vs all \
customers changes the result);
(5) joins follow the documented relationships listed above; an INNER JOIN \
that silently drops rows (e.g. customers with no orders) may be wrong when \
the question is about all rows of one side;
(6) completeness: the result must identify its entities (e.g. per-order rows \
should include the order id) and include the fields the question asks for;
(7) row limits: a LIMIT present in the SQL must be requested by the question \
(e.g. "top 5"); LIMIT must NOT be added when the question asks for all rows;
(8) proxy metrics: if the question asks for a metric that does not exist in \
the schema (e.g. a satisfaction score, forecast, or revenue definition \
requiring data that is not present), substituting a different metric is a \
REJECT, not an approximation.

Return JSON:
{{"verdict": "accept"|"reject"|"needs_review", "reasons": ["..."], "confidence": 0.0-1.0}}\
"""

CORRECTION_TEMPLATE = """\
The following Dremio SQL failed validation/execution.

Question: {question}
Failed SQL:
{sql}

Error (classified as {error_class}):
{error}

Schema:
{schema_text}

Produce a corrected version of the query that satisfies the question while \
remaining in Dremio's read-only SELECT/WITH syntax. Reply ONLY with the \
corrected SQL — no explanations.\
"""
