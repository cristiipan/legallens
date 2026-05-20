"""Per-worker task queue, backed by Redis BLPOP.

Each worker owns exactly one queue at key `queue:<worker_id>`. The dispatcher
RPUSHes; the worker BLPOPs. This gives us:
  - Strict FIFO per worker.
  - Backpressure for free: if a worker is slow, its queue grows in Redis.
  - No "two workers stealing the same task" race — the ring guarantees only
    one worker is ever the owner, and BLPOP is atomic anyway.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import structlog

from legallens.coordination.sentinel_client import SentinelClient
from legallens.orchestration.dispatcher import Task, worker_queue_key

log = structlog.get_logger()


class WorkerTaskQueue:
    def __init__(self, client: SentinelClient, worker_id: str) -> None:
        self.client = client
        self.worker_id = worker_id
        self.key = worker_queue_key(worker_id)

    async def pop(self, timeout: float = 5.0) -> Task | None:
        """BLPOP one task. Returns None on timeout — caller decides whether
        to loop (usually yes) or shut down."""
        master = self.client.master()
        result = await master.blpop([self.key], timeout=timeout)
        if not result:
            return None
        _key, payload = result
        return Task.model_validate_json(payload)

    async def iter_tasks(self, timeout: float = 5.0) -> AsyncIterator[Task]:
        """Async iterator wrapping pop(). Cleaner ergonomics in worker.run()."""
        while True:
            task = await self.pop(timeout=timeout)
            if task is None:
                continue
            yield task

    async def depth(self) -> int:
        """For observability. Logged occasionally by the worker."""
        return await self.client.replica().llen(self.key)
