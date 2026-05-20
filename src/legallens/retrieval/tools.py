"""Tool implementations exposed to the Cohere agent.

The agent calls these by name (see prompts/tools.py for definitions). They
do real I/O: PostgreSQL queries through the async session helper, vector
queries through the pluggable VectorStore.

Each tool function is its own coroutine + validates its own inputs. The
dispatcher (`dispatch_tool`) just routes — it does not handle errors. We
want exceptions to bubble up to the agent loop so the model gets a real
error message in the tool_result and can decide whether to retry.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog
from sqlalchemy import text

from legallens.db import session
from legallens.ingestion.embedder import Embedder, make_embedder
from legallens.ingestion.pinecone_writer import VectorStore, make_vector_store

log = structlog.get_logger()


# We cache these because they're expensive to construct (sentence-transformers
# loads a 200MB model on first call). Per-process singletons are fine.
@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    return make_embedder()


@lru_cache(maxsize=1)
def _vector_store() -> VectorStore:
    emb = _embedder()
    return make_vector_store(dimension=emb.dimension)


# --- Tools ----------------------------------------------------------------

async def extract_clauses(contract_id: str, category: str) -> dict[str, Any]:
    """All clauses of a given category from one contract."""
    log.info("tool.extract_clauses", contract_id=contract_id, category=category)

    async with session() as s:
        rows = await s.execute(
            text(
                """
                SELECT clause_id, text, page, risk_level, risk_rationale
                FROM clauses
                WHERE contract_id = :cid AND category = :cat
                ORDER BY clause_id
                """
            ),
            {"cid": contract_id, "cat": category},
        )
        clauses = [dict(r._mapping) for r in rows]

    return {"clauses": clauses, "count": len(clauses)}


async def search_similar_clauses(query: str, top_k: int = 5) -> dict[str, Any]:
    """Semantic search across the clause embedding index.

    Steps:
      1. Embed the query.
      2. Pinecone (or local) top-k by cosine.
      3. Hydrate text/category from PostgreSQL via clause_id.
    """
    log.info("tool.search_similar", q=query[:60], top_k=top_k)

    embedder = _embedder()
    store = _vector_store()

    [q_vec] = await embedder.embed([query])
    matches = await store.query(vector=q_vec, top_k=top_k)

    if not matches:
        return {"results": []}

    ids = [m["id"] for m in matches]
    async with session() as s:
        rows = await s.execute(
            text(
                """
                SELECT clause_id, contract_id, category, text
                FROM clauses
                WHERE clause_id = ANY(:ids)
                """
            ),
            {"ids": ids},
        )
        by_id = {r._mapping["clause_id"]: dict(r._mapping) for r in rows}

    results = []
    for m in matches:
        hydrated = by_id.get(m["id"])
        if not hydrated:
            continue
        results.append({**hydrated, "score": m["score"]})
    return {"results": results}


async def score_clause_risk(clause_id: str) -> dict[str, Any]:
    """Score a clause low/medium/high with rationale.

    First-tier implementation: deterministic heuristic on the clause text
    (length, presence of risk keywords). Good enough to bootstrap; replace
    with a focused LLM call once the agent loop is exercised end-to-end.
    """
    log.info("tool.score_risk", clause_id=clause_id)

    async with session() as s:
        row = (
            await s.execute(
                text(
                    "SELECT clause_id, category, text FROM clauses WHERE clause_id = :id"
                ),
                {"id": clause_id},
            )
        ).first()

    if not row:
        return {"error": f"clause not found: {clause_id}"}

    text_lower = row._mapping["text"].lower()
    category = row._mapping["category"]

    high_signal_terms = (
        "unlimited liability", "perpetual", "exclusive", "non-compete",
        "indemnify", "irrevocable", "without limitation",
    )
    medium_signal_terms = ("terminate", "breach", "damages", "liability")

    if any(t in text_lower for t in high_signal_terms):
        level, rationale = "high", (
            "Contains high-risk language (e.g. unlimited liability / non-compete / "
            "perpetual obligation). Worth a senior lawyer's review."
        )
    elif any(t in text_lower for t in medium_signal_terms):
        level, rationale = "medium", (
            "Standard but consequential clause — verify the specific terms align "
            "with our usual position."
        )
    else:
        level, rationale = "low", "Routine language; no obvious red flags."

    # Cache back to PG so subsequent calls are free.
    async with session() as s:
        await s.execute(
            text(
                """
                UPDATE clauses
                SET risk_level = :lvl, risk_rationale = :why
                WHERE clause_id = :id
                """
            ),
            {"lvl": level, "why": rationale, "id": clause_id},
        )

    return {
        "clause_id": clause_id,
        "category": category,
        "risk_level": level,
        "rationale": rationale,
    }


# --- Dispatch -------------------------------------------------------------

_TOOL_REGISTRY = {
    "extract_clauses": extract_clauses,
    "search_similar_clauses": search_similar_clauses,
    "score_clause_risk": score_clause_risk,
}


async def dispatch_tool(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    if name not in _TOOL_REGISTRY:
        raise ValueError(f"unknown tool: {name}. available: {list(_TOOL_REGISTRY)}")
    return await _TOOL_REGISTRY[name](**parameters)
