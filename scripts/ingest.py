"""Ingest driver.

Walks the CUAD JSON, dispatches one `ingest_contract` task per contract
through the consistent-hash ring. Workers (started separately, see
ingest_worker.py) pull tasks from their own queue and do the actual work:
write the contract row, embed clauses, upsert vectors, insert clause rows.

The driver itself is small. The "distributed" part isn't here — it's that
the driver only enqueues, and any number of workers can be running.

Run after services are up:
    docker compose up -d
    python scripts/init_db.py     # apply init_db.sql
    python scripts/ingest.py      # this file
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import structlog
from rich.console import Console

from legallens.coordination.sentinel_client import SentinelClient
from legallens.coordination.worker_registry import WorkerRegistry
from legallens.ingestion.parser import parse_cuad
from legallens.orchestration.dispatcher import (
    NoWorkersAvailable,
    Task,
    TaskDispatcher,
)

log = structlog.get_logger()
console = Console()


async def main(cuad_json: Path, limit: int | None) -> None:
    client = SentinelClient.from_settings()
    if not await client.ping():
        raise SystemExit("Redis Sentinel unreachable — is docker compose up?")

    registry = WorkerRegistry(client)
    workers = await registry.refresh_ring()
    if not workers:
        raise SystemExit(
            "No live workers found. Start at least one with "
            "`python scripts/ingest_worker.py` and re-run."
        )

    dispatcher = TaskDispatcher(registry, client)

    enqueued = 0
    for parsed in parse_cuad(cuad_json):
        if limit and enqueued >= limit:
            break
        task = Task(
            kind="ingest_contract",
            routing_key=parsed.contract_id,
            payload={
                "contract_id": parsed.contract_id,
                "filename": parsed.filename,
                "clauses": [c.model_dump() for c in parsed.clauses],
            },
        )
        try:
            wid = await dispatcher.dispatch(task)
        except NoWorkersAvailable:
            console.print("[red]ring went empty mid-dispatch — aborting[/red]")
            return
        enqueued += 1
        if enqueued % 25 == 0:
            console.print(
                f"[cyan]enqueued {enqueued} contracts...[/cyan] "
                f"(last → {wid})"
            )

    console.print(f"[green]enqueued {enqueued} contracts across {len(workers)} workers[/green]")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dispatch CUAD contracts to ingest workers")
    p.add_argument(
        "--cuad-json",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "cuad" / "CUAD_v1.json",
        help="Path to the unzipped CUAD_v1.json",
    )
    p.add_argument("--limit", type=int, default=None, help="Stop after N contracts (for dev)")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    asyncio.run(main(args.cuad_json, args.limit))
