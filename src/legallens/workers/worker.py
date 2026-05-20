"""Worker process.

A worker:
  1. Registers itself with the WorkerRegistry (heartbeat key + join event).
  2. Spawns a heartbeat loop in the background.
  3. BLPOPs its task queue forever, dispatching by task.kind.

Why one process = one worker_id (not threads):
  - GIL-friendly: ingestion is largely I/O (HTTP to Cohere/Pinecone, PG
    writes), but embedding fallback uses numpy/torch which release the GIL.
    Process-per-worker keeps the model simple.
  - Clean blast radius: when a worker crashes, only its queue (and the
    contracts assigned to it via the ring) are affected.
  - Easy horizontal scale: `docker compose up --scale ingest-worker=N`.
"""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from legallens.coordination.sentinel_client import SentinelClient
from legallens.coordination.worker_registry import WorkerRegistry
from legallens.orchestration.dispatcher import Task
from legallens.workers.task_queue import WorkerTaskQueue

log = structlog.get_logger()


TaskHandler = Callable[[Task], Awaitable[dict[str, Any]]]


def default_worker_id() -> str:
    """`<hostname>-<pid>-<rand>` — stable enough to debug, unique per process."""
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


class Worker:
    """Long-running worker process. Construct, then `await run()`."""

    def __init__(
        self,
        worker_id: str | None = None,
        handlers: dict[str, TaskHandler] | None = None,
        client: SentinelClient | None = None,
    ) -> None:
        self.worker_id = worker_id or default_worker_id()
        self.handlers: dict[str, TaskHandler] = handlers or {}
        self.client = client or SentinelClient.from_settings()
        self.registry = WorkerRegistry(self.client)
        self.queue = WorkerTaskQueue(self.client, self.worker_id)
        self._stop = asyncio.Event()

    def register_handler(self, kind: str, handler: TaskHandler) -> None:
        self.handlers[kind] = handler

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                # Windows / some test envs — fall through, kill -9 still works.
                pass

    async def run(self) -> None:
        log.info("worker.starting", worker_id=self.worker_id, kinds=list(self.handlers))
        self._install_signal_handlers()
        await self.registry.register(self.worker_id)
        hb_task = asyncio.create_task(self.registry.heartbeat_loop(self.worker_id))

        try:
            while not self._stop.is_set():
                task = await self.queue.pop(timeout=2.0)
                if task is None:
                    continue
                await self._handle(task)
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
            log.info("worker.stopped", worker_id=self.worker_id)

    async def _handle(self, task: Task) -> None:
        handler = self.handlers.get(task.kind)
        if handler is None:
            log.error(
                "worker.no_handler",
                worker_id=self.worker_id,
                kind=task.kind,
                known=list(self.handlers),
            )
            return
        log.info("worker.task.start", worker_id=self.worker_id, kind=task.kind,
                 routing_key=task.routing_key)
        try:
            await handler(task)
        except Exception:
            log.exception(
                "worker.task.failed",
                worker_id=self.worker_id,
                kind=task.kind,
                routing_key=task.routing_key,
            )
        else:
            log.info("worker.task.done", worker_id=self.worker_id, kind=task.kind,
                     routing_key=task.routing_key)
