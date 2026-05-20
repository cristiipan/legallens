"""Heartbeat monitor — the *consumer* side of worker heartbeats.

The producer side (a worker setting its own `worker:hb:<id>` TTL) lives in
WorkerRegistry.heartbeat_loop. This module is what runs on the dispatcher
side to *react* to a dead worker: scan the heartbeat keys on an interval,
diff against the local ring, and trigger a ring rebuild + re-dispatch of
any tasks the dead worker had queued.

Pub/sub catches joins quickly but doesn't catch deaths (no event fires
when a TTL expires). So we still need this periodic scan.
"""
from __future__ import annotations

import asyncio

import structlog

from legallens.coordination.worker_registry import WorkerRegistry

log = structlog.get_logger()


class HeartbeatMonitor:
    """Periodically reconciles the ring with the authoritative heartbeat set."""

    def __init__(
        self,
        registry: WorkerRegistry,
        scan_interval_seconds: float = 5.0,
    ) -> None:
        self.registry = registry
        self.interval = scan_interval_seconds

    async def run(self, on_change: callable | None = None) -> None:
        """Run forever. Cancel the task to stop.

        on_change is called with the new worker list whenever the ring
        membership changes. Use it to trigger re-dispatch of orphaned tasks.
        """
        last_workers: list[str] = []
        try:
            while True:
                workers = await self.registry.refresh_ring()
                if workers != last_workers:
                    log.info(
                        "heartbeat.membership_changed",
                        added=sorted(set(workers) - set(last_workers)),
                        removed=sorted(set(last_workers) - set(workers)),
                    )
                    last_workers = workers
                    if on_change:
                        result = on_change(workers)
                        if asyncio.iscoroutine(result):
                            await result
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            log.info("heartbeat.stopped")
            raise
