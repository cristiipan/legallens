"""Vector index writer.

Two backends:
  - PineconeIndex: managed serverless index, what we use in prod.
  - LocalNumpyIndex: a single .npy + .json on disk. Slower at scale but lets
    the project boot without a Pinecone account.

Both expose the same write/query surface. The agent's `search_similar_clauses`
tool only knows about VectorStore, so swapping backends is one config flip.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import structlog

from legallens.config import get_settings

log = structlog.get_logger()


class VectorRecord(Protocol):
    id: str
    vector: list[float]
    metadata: dict[str, Any]


class VectorStore(Protocol):
    async def upsert(self, records: list[dict[str, Any]]) -> None:
        """records = [{'id': str, 'values': [float], 'metadata': {...}}]"""
        ...

    async def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns [{'id', 'score', 'metadata'}], highest score first."""
        ...


class PineconeIndex:
    """Pinecone backend. We use serverless because (a) free tier, (b) we
    don't want to manage pod-based scaling for 15K vectors."""

    def __init__(self, index_name: str, dimension: int) -> None:
        from pinecone import Pinecone, ServerlessSpec  # heavy import

        s = get_settings()
        pc = Pinecone(api_key=s.pinecone_api_key)
        if index_name not in [i.name for i in pc.list_indexes()]:
            log.info("pinecone.creating_index", name=index_name, dim=dimension)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = pc.Index(index_name)

    async def upsert(self, records: list[dict[str, Any]]) -> None:
        # pinecone-python is sync; one call per batch is fine for ingest.
        self.index.upsert(vectors=records)

    async def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        resp = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter or None,
        )
        return [
            {"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})}
            for m in resp.get("matches", [])
        ]


class LocalNumpyIndex:
    """Single-file local vector store. Adequate for 500 contracts × ~30 clauses.

    On-disk layout:
      <root>/vectors.npy   — (N, D) float32 array of vectors
      <root>/manifest.json — list of {id, metadata}, parallel to rows of vectors.npy
    """

    def __init__(self, root: Path, dimension: int) -> None:
        import numpy as np  # local import to keep numpy out of the import path

        self.np = np
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.root / "vectors.npy"
        self.manifest_path = self.root / "manifest.json"
        self.dimension = dimension

        if self.vectors_path.exists() and self.manifest_path.exists():
            self.vectors = np.load(self.vectors_path)
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            self.vectors = np.zeros((0, dimension), dtype="float32")
            self.manifest = []

    def _persist(self) -> None:
        self.np.save(self.vectors_path, self.vectors)
        self.manifest_path.write_text(json.dumps(self.manifest))

    async def upsert(self, records: list[dict[str, Any]]) -> None:
        np = self.np
        ids_existing = {m["id"]: i for i, m in enumerate(self.manifest)}
        new_vecs: list[list[float]] = []
        new_entries: list[dict[str, Any]] = []

        for r in records:
            vec = r["values"]
            entry = {"id": r["id"], "metadata": r.get("metadata", {})}
            if r["id"] in ids_existing:
                idx = ids_existing[r["id"]]
                self.vectors[idx] = np.array(vec, dtype="float32")
                self.manifest[idx] = entry
            else:
                new_vecs.append(vec)
                new_entries.append(entry)

        if new_vecs:
            block = np.array(new_vecs, dtype="float32")
            self.vectors = np.concatenate([self.vectors, block], axis=0)
            self.manifest.extend(new_entries)

        self._persist()

    async def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        np = self.np
        if len(self.manifest) == 0:
            return []
        q = np.array(vector, dtype="float32")
        # Cosine sim, assuming both q and stored vectors are L2-normalized
        # (the local embedder normalizes; the Cohere one does too).
        scores = self.vectors @ q
        # Apply optional metadata filter — naive scan, fine at this scale.
        candidate_idx = list(range(len(self.manifest)))
        if filter:
            candidate_idx = [
                i for i in candidate_idx
                if all(self.manifest[i]["metadata"].get(k) == v for k, v in filter.items())
            ]
        ranked = sorted(candidate_idx, key=lambda i: -float(scores[i]))[:top_k]
        return [
            {
                "id": self.manifest[i]["id"],
                "score": float(scores[i]),
                "metadata": self.manifest[i]["metadata"],
            }
            for i in ranked
        ]


def make_vector_store(dimension: int) -> VectorStore:
    s = get_settings()
    key = (s.pinecone_api_key or "").strip()
    if key and not key.startswith("your_") and key != "mock":
        return PineconeIndex(index_name=s.pinecone_index_name, dimension=dimension)
    log.info("vector_store.using_local_fallback")
    return LocalNumpyIndex(root=Path(s.local_vector_root), dimension=dimension)
