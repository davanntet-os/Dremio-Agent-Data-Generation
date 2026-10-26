"""dataagent command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from dataagent.config import get_settings

console = Console()


def cmd_discover(args: argparse.Namespace) -> int:
    from dataagent.dremio.client import DremioClient
    from dataagent.schema.discovery import SchemaDiscoveryService

    with DremioClient() as client:
        service = SchemaDiscoveryService(client)
        console.print("[bold]Discovering Dremio catalog...[/bold]")
        counters = service.discover(include_samples=not args.no_samples)

    table = Table(title="Schema discovery summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for key in sorted(counters):
        table.add_row(key, str(counters[key]))
    console.print(table)

    from dataagent.schema.store import SchemaStore

    store = SchemaStore()
    datasets = store.list_datasets()
    for ds in datasets:
        cols = ", ".join(f"{c['name']} {c['type']}" for c in ds["columns"][:12])
        more = f" (+{len(ds['columns']) - 12} more)" if len(ds["columns"]) > 12 else ""
        console.print(f"[cyan]{ds['name']}[/cyan] ({ds['dataset_type']}): {cols}{more}")
    store.close()
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from dataagent.dremio.client import DremioClient
    from dataagent.validation.parser import ParserValidator

    settings = get_settings()
    sql = args.sql.rstrip().rstrip(";")
    validator = ParserValidator(max_rows=settings.query_max_rows)
    verdict = validator.parse_and_validate(sql)
    if not verdict.ok:
        console.print(f"[red]Rejected: {verdict.reason}[/red]")
        return 1
    with DremioClient() as client:
        result = client.execute_sql(sql, timeout_seconds=settings.query_timeout_seconds)
    if result.state != "COMPLETED":
        console.print(f"[red]Job {result.job_id} ({result.state}): {result.error_message}[/red]")
        return 1
    console.print(f"[green]Rows: {len(result.rows)} (job {result.job_id})[/green]")
    if result.rows:
        table = Table()
        for col in result.columns:
            table.add_column(col)
        for row in result.rows[: min(args.max_display, len(result.rows))]:
            table.add_row(*[str(row.get(c)) for c in result.columns])
        console.print(table)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from dataagent.pipeline import DataGenerationPipeline

    settings = get_settings()
    if args.workers:
        settings.pipeline_workers = args.workers
    pipeline = DataGenerationPipeline(settings)
    outputs = pipeline.run(target=args.target, fresh=args.fresh)
    console.print("[bold green]Generation complete[/bold green]")
    for key, value in outputs.items():
        console.print(f"  {key}: {value}")
    return 0


def cmd_llm_check(_: argparse.Namespace) -> int:
    from dataagent.llm.base import ChatMessage
    from dataagent.llm.glm import build_model_client

    settings = get_settings()
    console.print(f"Mode: {settings.glm_mode}")
    client = build_model_client(settings)
    if hasattr(client, "verify_provider"):
        ids = client.verify_provider()
        glm_models = sorted(m for m in ids if "glm" in m)
        console.print(f"Served GLM models: {', '.join(glm_models)}")
    resp = client.complete(
        [ChatMessage("user", "Reply with exactly: OK")],
        temperature=0.1, max_tokens=32)
    console.print(f"Model id: {resp.model}")
    console.print(f"Reply: {resp.content[:80]}")
    client.close()
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    settings = get_settings()
    console.print(f"dataagent 0.1.0 — GLM model={settings.glm_model} (mode={settings.glm_mode}), "
                  f"dremio={settings.dremio_uri}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataagent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_disc = sub.add_parser("discover", help="Walk the Dremio catalog into the schema cache")
    p_disc.add_argument("--no-samples", action="store_true", help="Skip sample/sample-file datasets")
    p_disc.set_defaults(func=cmd_discover)

    p_q = sub.add_parser("query", help="Run a read-only SQL query (smoke test)")
    p_q.add_argument("sql")
    p_q.add_argument("--max-display", type=int, default=10)
    p_q.set_defaults(func=cmd_query)

    p_g = sub.add_parser("generate", help="Generate, validate, verify and export SFT examples")
    p_g.add_argument("--target", type=int, default=500, help="Number of candidate questions")
    p_g.add_argument("--workers", type=int, default=None,
                     help="Parallel workers (default: PIPELINE_WORKERS from env)")
    p_g.add_argument("--fresh", action="store_true",
                     help="Reset the cumulative example library (start over)")
    p_g.set_defaults(func=cmd_generate)

    p_v = sub.add_parser("version", help="Show version/config summary")
    p_v.set_defaults(func=cmd_version)

    p_c = sub.add_parser("llm-check", help="Verify LLM provider endpoint and model id")
    p_c.set_defaults(func=cmd_llm_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        console.print(f"[red]error:[/red] {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
