"""Clause embedder.

Two backends with the same interface:
  - CohereEmbedder: `cohere.AsyncClient.embed(model="embed-english-v3.0")`
  - LocalEmbedder:  sentence-transformers (`all-MiniLM-L6-v2`), runs offline.

The local backend is what we use when no Cohere key is configured. It's
slower per call but has no rate limit and works without an account, which
makes the project clonable + runnable by reviewers.

Vectors from the two backends are NOT interchangeable — they have different
dimensions and live in different geometric spaces. If you switch backends,
re-embed everything.
"""
from __future__ import annotations

from typing import Protocol

import structlog

from legallens.config import get_settings

log = structlog.get_logger()


class Embedder(Protocol):
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class CohereEmbedder:
    """Production path. Uses Cohere's embed-english-v3.0 (1024-dim)."""

    dimension = 1024

    def __init__(self) -> None:
        import cohere  # local import — only needed on this path

        s = get_settings()
        self.client = cohere.AsyncClient(api_key=s.cohere_api_key)
        self.model = "embed-english-v3.0"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Cohere allows up to 96 texts per call.
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):
            chunk = texts[i : i + 96]
            resp = await self.client.embed(
                texts=chunk,
                model=self.model,
                input_type="search_document",
            )
            out.extend(resp.embeddings)
        return out


class LocalEmbedder:
    """Offline fallback. Lazy-loads sentence-transformers so importing this
    module doesn't pay the 200MB model download cost unless you call embed()."""

    dimension = 384  # all-MiniLM-L6-v2

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import

            log.info("embedder.local.loading", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        # sentence-transformers is sync; for ingest throughput we'd push this
        # to a thread pool, but here the worker is already isolated per-process
        # so blocking the event loop briefly is fine.
        return model.encode(texts, normalize_embeddings=True).tolist()


def make_embedder() -> Embedder:
    """Pick backend by config. Cohere if a real key is set, else local."""
    s = get_settings()
    key = (s.cohere_api_key or "").strip()
    if key and not key.startswith("your_") and key != "mock":
        return CohereEmbedder()
    log.info("embedder.using_local_fallback")
    return LocalEmbedder()
