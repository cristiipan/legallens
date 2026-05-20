"""Ingest worker entrypoint.

Each invocation is one worker process. Scale horizontally with
    docker compose up --scale ingest-worker=N
or run multiple terminals locally:
    python scripts/ingest_worker.py
    python scripts/ingest_worker.py
    ...

The handler below is the actual work a worker does when it pops an
`ingest_contract` task: write the contract row, embed clauses, upsert
vectors, insert clause rows. Each step is idempotent (ON CONFLICT DO
NOTHING / UPSERTs) so a retried task is safe.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import text

from legallens.db import session
from legallens.ingestion.embedder import make_embedder
from legallens.ingestion.pinecone_writer import make_vector_store
from legallens.orchestration.dispatcher import Task
from legallens.workers.worker import Worker

log = structlog.get_logger()


_embedder = None
_vector_store = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = make_embedder()
    return _embedder


def _get_store():
    global _vector_store
    if _vector_store is None:
        emb = _get_embedder()
        _vector_store = make_vector_store(dimension=emb.dimension)
    return _vector_store


async def handle_ingest_contract(task: Task) -> dict[str, Any]:
    payload = task.payload
    contract_id = payload["contract_id"]
    filename = payload["filename"]
    clauses = payload["clauses"]

    # 1. Upsert contract row.
    async with session() as s:
        await s.execute(
            text(
                """
                INSERT INTO contracts (contract_id, filename, ingested_at)
                VALUES (:cid, :fn, NOW())
                ON CONFLICT (contract_id) DO UPDATE
                SET filename = EXCLUDED.filename, ingested_at = NOW()
                """
            ),
            {"cid": contract_id, "fn": filename},
        )

    if not clauses:
        log.info("worker.ingest.empty_contract", contract_id=contract_id)
        return {"contract_id": contract_id, "n_clauses": 0}

    # 2. Embed clause texts.
    texts = [c["text"] for c in clauses]
    vectors = await _get_embedder().embed(texts)

    # 3. Upsert vectors with metadata.
    store = _get_store()
    records = [
        {
            "id": c["clause_id"],
            "values": v,
            "metadata": {
                "contract_id": contract_id,
                "category": c["category"],
            },
        }
        for c, v in zip(clauses, vectors)
    ]
    await store.upsert(records)

    # 4. Insert clause rows.
    now = datetime.utcnow()
    async with session() as s:
        for c in clauses:
            await s.execute(
                text(
                    """
                    INSERT INTO clauses
                        (clause_id, contract_id, category, text, vector_id, embedded_at)
                    VALUES (:cid, :ctr, :cat, :txt, :vid, :emb)
                    ON CONFLICT (contract_id, clause_id) DO UPDATE
                    SET text = EXCLUDED.text,
                        vector_id = EXCLUDED.vector_id,
                        embedded_at = EXCLUDED.embedded_at
                    """
                ),
                {
                    "cid": c["clause_id"],
                    "ctr": contract_id,
                    "cat": c["category"],
                    "txt": c["text"],
                    "vid": c["clause_id"],
                    "emb": now,
                },
            )

    log.info("worker.ingest.done", contract_id=contract_id, n_clauses=len(clauses))
    return {"contract_id": contract_id, "n_clauses": len(clauses)}


async def main() -> None:
    worker = Worker()
    worker.register_handler("ingest_contract", handle_ingest_contract)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
