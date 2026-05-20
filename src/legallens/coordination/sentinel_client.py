"""Thin wrapper around redis-py's async Sentinel.

Why Sentinel instead of plain Redis?
  - The orchestration layer (dispatcher, registry, worker queues) cannot
    afford the Redis master being a single point of failure. If the master
    dies, Sentinel promotes a replica within ~10s and clients reconnect to
    the new master.
  - We don't go all the way to Redis Cluster because we don't need sharding
    at this scale (500 contracts, ~15K clauses); we need availability.

This module hides the Sentinel <-> client plumbing so the rest of the code
just does `client = await get_master()` and gets a working connection.
"""
from __future__ import annotations

import structlog
from redis.asyncio import Redis
from redis.asyncio.sentinel import Sentinel

from legallens.config import get_settings

log = structlog.get_logger()


class SentinelClient:
    """Lazily resolves a master/replica Redis connection via Sentinel.

    Hold one of these per process. The underlying redis-py Sentinel object
    maintains a small connection pool internally.
    """

    def __init__(
        self,
        sentinels: list[tuple[str, int]],
        master_name: str,
        socket_timeout: float = 0.5,
    ) -> None:
        self.sentinel = Sentinel(sentinels, socket_timeout=socket_timeout)
        self.master_name = master_name

    @classmethod
    def from_settings(cls) -> SentinelClient:
        s = get_settings()
        sentinels = [
            tuple(h.split(":")) for h in s.redis_sentinels.split(",") if h.strip()
        ]
        # Sentinel ports default to 26379 if not specified.
        sentinels = [(h, int(p) if p else 26379) for h, p in sentinels]
        return cls(sentinels=sentinels, master_name=s.redis_master_name)

    def master(self) -> Redis:
        """Master connection for writes. Auto-fails over to the new master
        when Sentinel promotes one."""
        return self.sentinel.master_for(self.master_name, decode_responses=True)

    def replica(self) -> Redis:
        """Read-only replica connection. Useful for SCAN-heavy reads (the
        registry's ring rebuild does a lot of these)."""
        return self.sentinel.slave_for(self.master_name, decode_responses=True)

    async def ping(self) -> bool:
        try:
            return await self.master().ping()
        except Exception:
            log.exception("sentinel.ping_failed")
            return False
