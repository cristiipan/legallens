"""Smoke tests for the consistent hash ring.

The two properties we care about:
  1. Same key -> same worker, deterministically.
  2. Adding a worker only reassigns ~K/N of the keys (no full reshuffle).
"""
from __future__ import annotations

from collections import Counter

import pytest

from legallens.workers.hash_ring import ConsistentHashRing


def test_empty_ring_returns_none():
    ring = ConsistentHashRing()
    assert ring.get_worker("contract-1") is None


def test_deterministic_routing():
    ring = ConsistentHashRing()
    ring.add_worker("w1")
    ring.add_worker("w2")
    ring.add_worker("w3")
    # Same key always lands on the same worker.
    assignments = {f"c-{i}": ring.get_worker(f"c-{i}") for i in range(100)}
    for k, v in assignments.items():
        assert ring.get_worker(k) == v


def test_distribution_is_reasonably_even():
    ring = ConsistentHashRing(virtual_nodes=200)
    for w in ["w1", "w2", "w3", "w4"]:
        ring.add_worker(w)
    counts = Counter(ring.get_worker(f"key-{i}") for i in range(10_000))
    # With 200 vnodes/worker, no worker should own more than ~40% of keys.
    for w in counts:
        share = counts[w] / 10_000
        assert 0.10 < share < 0.40, f"{w} got {share:.2%} of keys"


def test_add_worker_minimally_reshuffles():
    ring = ConsistentHashRing(virtual_nodes=200)
    for w in ["w1", "w2", "w3"]:
        ring.add_worker(w)
    before = {f"k-{i}": ring.get_worker(f"k-{i}") for i in range(2_000)}

    ring.add_worker("w4")
    after = {f"k-{i}": ring.get_worker(f"k-{i}") for i in range(2_000)}

    moved = sum(1 for k in before if before[k] != after[k])
    # With 4 workers, ideally ~1/4 of keys move. Allow [0.10, 0.45].
    fraction = moved / len(before)
    assert 0.10 < fraction < 0.45, f"unexpected reshuffle: {fraction:.2%}"


def test_remove_worker_minimally_reshuffles():
    ring = ConsistentHashRing(virtual_nodes=200)
    for w in ["w1", "w2", "w3", "w4"]:
        ring.add_worker(w)
    before = {f"k-{i}": ring.get_worker(f"k-{i}") for i in range(2_000)}

    ring.remove_worker("w4")
    after = {f"k-{i}": ring.get_worker(f"k-{i}") for i in range(2_000)}

    # Keys NOT owned by w4 before should remain on the same worker after.
    for k, w in before.items():
        if w != "w4":
            assert after[k] == w, f"{k} unexpectedly moved from {w} to {after[k]}"


def test_replace_workers_atomicity():
    ring = ConsistentHashRing()
    ring.add_worker("w1")
    ring.add_worker("w2")
    ring.replace_workers(["w3", "w4", "w5"])
    assert set(ring.workers()) == {"w3", "w4", "w5"}
    assert ring.get_worker("anything") in {"w3", "w4", "w5"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
