"""Redis pub/sub helpers.

Two channels in this project:
  - worker:events     — cluster membership (join/leave). Consumed by registry.
  - agent:stream:<id> — per-review event stream from worker -> API.
                        The API's SSE handler subscribes and forwards.

Using pub/sub for the agent stream means the worker that's actually running
the agent can be on a different host than the API process holding the
client's SSE connection. Without pub/sub we'd need to pin reviews to the
process that received the HTTP request, which defeats the whole point of
having a worker pool.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import structlog

from legallens.coordination.sentinel_client import SentinelClient

log = structlog.get_logger()


def agent_stream_channel(stream_id: str) -> str:
    return f"agent:stream:{stream_id}"


async def publish_event(
    client: SentinelClient,
    channel: str,
    payload: dict[str, Any],
) -> None:
    master = client.master()
    await master.publish(channel, json.dumps(payload))


async def subscribe(
    client: SentinelClient,
    channel: str,
) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded JSON payloads from `channel` until the caller closes it.

    The pubsub object is created here, not by the caller; we own its lifecycle.
    """
    pubsub = client.master().pubsub()
    await pubsub.subscribe(channel)
    try:
        async for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            data = msg.get("data")
            if not data:
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                log.warning("pubsub.bad_payload", channel=channel, data=data[:200])
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()
