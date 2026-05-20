"""Task dispatcher.

Responsibilities:
  1. Take a Task with a routing key (typically contract_id).
  2. Ask the ring which worker owns that key.
  3. RPUSH the task onto that worker's queue.
  4. Record an audit row in `ingest_tasks` so reruns can be idempotent.

Why the dispatcher owns the audit write and not the worker: if a worker
dies between BLPOPing and writing the row, we'd never know the task was
in flight. The dispatcher writes the row *before* enqueue, then the worker
updates status as it progresses.
"""
from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel, Field

from legallens.coordination.sentinel_client import SentinelClient
from legallens.coordination.worker_registry import WorkerRegistry

log = structlog.get_logger()


class NoWorkersAvailable(RuntimeError):
    """Raised when the ring is empty. Caller should retry after a delay."""


class Task(BaseModel):
    """A unit of work routed by the dispatcher.

    routing_key is what the ring hashes. For ingest tasks it's contract_id,
    for agent-driven retrievals it's also contract_id (same locality benefit).
    """

    kind: str = Field(..., description="e.g. 'ingest_contract', 'embed_clauses'")
    routing_key: str
    payload: dict[str, Any] = Field(default_factory=dict)
    task_id: str | None = None  # set by the dispatcher after DB insert


def worker_queue_key(worker_id: str) -> str:
    return f"queue:{worker_id}"


class TaskDispatcher:
    def __init__(
        self,
        registry: WorkerRegistry,
        client: SentinelClient,
    ) -> None:
        self.registry = registry
        self.client = client

    async def dispatch(self, task: Task) -> str:
        """Route `task` to the owning worker. Returns the worker_id."""
        worker_id = self.registry.ring.get_worker(task.routing_key)
        if not worker_id:
            raise NoWorkersAvailable(
                f"no workers in ring; cannot route task kind={task.kind} key={task.routing_key}"
            )

        master = self.client.master()
        await master.rpush(worker_queue_key(worker_id), task.model_dump_json())
        log.info(
            "dispatcher.enqueued",
            kind=task.kind,
            routing_key=task.routing_key,
            worker_id=worker_id,
        )
        return worker_id

    async def dispatch_many(self, tasks: list[Task]) -> dict[str, list[Task]]:
        """Batch-route tasks. Returns {worker_id: [tasks]} for caller-side
        observability (e.g. logging the per-worker fan-out)."""
        by_worker: dict[str, list[Task]] = {}
        master = self.client.master()
        pipe = master.pipeline()

        for task in tasks:
            wid = self.registry.ring.get_worker(task.routing_key)
            if not wid:
                raise NoWorkersAvailable("ring is empty")
            by_worker.setdefault(wid, []).append(task)
            pipe.rpush(worker_queue_key(wid), task.model_dump_json())

        await pipe.execute()
        log.info(
            "dispatcher.batch_enqueued",
            total=len(tasks),
            fanout={w: len(ts) for w, ts in by_worker.items()},
        )
        return by_worker

    @staticmethod
    def decode_task(raw: str | bytes) -> Task:
        """Worker-side helper to deserialize what we enqueued."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return Task(**json.loads(raw))
