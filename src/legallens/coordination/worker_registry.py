"""Worker registry built on Redis TTL keys + pub/sub.

Lifecycle of a worker:
  1. On startup, worker calls register(): sets `worker:hb:<id>` with TTL,
     publishes `join:<id>` on `worker:events`.
  2. Every HEARTBEAT_INTERVAL seconds, worker refreshes the TTL.
  3. If the worker dies, the TTL expires (after HB_TTL seconds), and the
     next registry scan removes it from the ring.
  4. The registry also subscribes to `worker:events` so joins propagate
     immediately, without waiting for the next scan.

The registry is intentionally simple — no ZAB, no Raft. Redis already gives
us a consistent view per-key, and Sentinel handles master failover. The cost
is that during a Sentinel-driven failover (~5-10s) registrations may briefly
be invisible; we live with that.
"""
from __future__ import annotations

import asyncio

import structlog

from legallens.coordination.sentinel_client import SentinelClient
from legallens.workers.hash_ring import ConsistentHashRing

log = structlog.get_logger()

HB_KEY_PREFIX = "worker:hb:"
HB_TTL_SECONDS = 10
HEARTBEAT_INTERVAL = 3
EVENTS_CHANNEL = "worker:events"


class WorkerRegistry:
    """Holds the local view of which workers are alive + a hash ring of them.

    Two ways the local view stays fresh:
      - subscribe(): pub/sub on `worker:events` (low latency, lossy)
      - refresh_ring(): periodic SCAN of `worker:hb:*` (high latency, authoritative)
    Run both. Pub/sub catches joins fast; the scan catches missed events and
    expired keys.
    """

    def __init__(self, client: SentinelClient) -> None:
        self.client = client
        self.ring = ConsistentHashRing()

    async def register(self, worker_id: str) -> None:
        """Called by a worker process at startup."""
        master = self.client.master()
        await master.set(f"{HB_KEY_PREFIX}{worker_id}", "1", ex=HB_TTL_SECONDS)
        await master.publish(EVENTS_CHANNEL, f"join:{worker_id}")
        log.info("registry.register", worker_id=worker_id)

    async def heartbeat(self, worker_id: str) -> None:
        master = self.client.master()
        await master.set(f"{HB_KEY_PREFIX}{worker_id}", "1", ex=HB_TTL_SECONDS)

    async def deregister(self, worker_id: str) -> None:
        master = self.client.master()
        await master.delete(f"{HB_KEY_PREFIX}{worker_id}")
        await master.publish(EVENTS_CHANNEL, f"leave:{worker_id}")
        log.info("registry.deregister", worker_id=worker_id)

    async def heartbeat_loop(self, worker_id: str) -> None:
        """Run forever; refreshes the heartbeat key. Workers spawn this as a
        background task and let it die with the process."""
        try:
            while True:
                await self.heartbeat(worker_id)
                await asyncio.sleep(HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            await self.deregister(worker_id)
            raise

    async def list_workers(self) -> list[str]:
        """Authoritative scan of live worker keys."""
        replica = self.client.replica()
        workers: list[str] = []
        async for key in replica.scan_iter(match=f"{HB_KEY_PREFIX}*"):
            workers.append(key[len(HB_KEY_PREFIX):])
        return sorted(workers)

    async def refresh_ring(self) -> list[str]:
        """Rebuild the local hash ring from the authoritative worker set.

        Returns the new worker list so callers can log/diff.
        """
        workers = await self.list_workers()
        self.ring.replace_workers(workers)
        log.info("registry.ring_refreshed", workers=workers)
        return workers

    async def watch_events(self, on_change: callable | None = None) -> None:
        """Subscribe to `worker:events` and refresh the ring on each message.

        Run this as a background task in the dispatcher process. `on_change`
        is invoked (sync or async) after every refresh; useful for triggering
        a re-dispatch of in-flight tasks whose owner changed.
        """
        pubsub = self.client.master().pubsub()
        await pubsub.subscribe(EVENTS_CHANNEL)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                log.info("registry.event", payload=msg.get("data"))
                workers = await self.refresh_ring()
                if on_change:
                    result = on_change(workers)
                    if asyncio.iscoroutine(result):
                        await result
        finally:
            await pubsub.unsubscribe(EVENTS_CHANNEL)
            await pubsub.close()
