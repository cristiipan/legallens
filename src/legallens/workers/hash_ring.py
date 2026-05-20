"""Consistent hash ring for routing contract tasks to workers.

Why consistent hashing instead of round-robin or random?
  - Task locality: every task for the same contract_id lands on the same
    worker, so that worker can cache the parsed contract, reuse open DB
    connections to the same partition, and avoid double-embedding clauses.
  - Smooth rebalancing: when a worker joins or leaves, only ~K/N of the keys
    move (K = total keys, N = workers), instead of every key being remapped.
  - No central coordinator on the hot path: the ring is replicated in every
    process; lookups are O(log N) and require no network round-trip.

We use virtual nodes (150 per worker by default) to smooth out the
key distribution. Without virtual nodes, a 3-worker ring leaves big gaps
where one worker can own 60%+ of the keyspace by chance.
"""
from __future__ import annotations

import bisect
import hashlib
from collections.abc import Iterable


class ConsistentHashRing:
    """Thread-unsafe consistent hash ring. Wrap in a lock if mutated concurrently.

    Designed for read-heavy use: get_worker() is called per dispatched task,
    add_worker / remove_worker only on cluster membership changes.
    """

    def __init__(self, virtual_nodes: int = 150) -> None:
        self.virtual_nodes = virtual_nodes
        self._ring: dict[int, str] = {}
        self._sorted_hashes: list[int] = []

    @staticmethod
    def _hash(key: str) -> int:
        # MD5 is fine here — we want a uniform 128-bit distribution, not a
        # cryptographic guarantee. SHA-256 would work too; MD5 is just faster.
        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)

    def add_worker(self, worker_id: str) -> None:
        for i in range(self.virtual_nodes):
            h = self._hash(f"{worker_id}#{i}")
            self._ring[h] = worker_id
        self._sorted_hashes = sorted(self._ring.keys())

    def remove_worker(self, worker_id: str) -> None:
        for i in range(self.virtual_nodes):
            h = self._hash(f"{worker_id}#{i}")
            self._ring.pop(h, None)
        self._sorted_hashes = sorted(self._ring.keys())

    def replace_workers(self, worker_ids: Iterable[str]) -> None:
        """Atomic-from-the-caller's-POV rebuild. Used by the registry when it
        notices the live worker set has drifted from what the ring believes.
        """
        self._ring.clear()
        for wid in worker_ids:
            for i in range(self.virtual_nodes):
                h = self._hash(f"{wid}#{i}")
                self._ring[h] = wid
        self._sorted_hashes = sorted(self._ring.keys())

    def get_worker(self, key: str) -> str | None:
        if not self._sorted_hashes:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self._sorted_hashes, h)
        # Wrap around the ring if we fell off the end.
        if idx == len(self._sorted_hashes):
            idx = 0
        return self._ring[self._sorted_hashes[idx]]

    def workers(self) -> list[str]:
        return sorted(set(self._ring.values()))

    def __len__(self) -> int:
        return len(set(self._ring.values()))

    def __repr__(self) -> str:
        return f"ConsistentHashRing(workers={self.workers()}, vnodes={self.virtual_nodes})"
