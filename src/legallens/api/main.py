"""FastAPI entry point.

Two main endpoints:
  - POST /review     → start a contract review, stream events via SSE
  - POST /followup   → ask a follow-up question, stream answer via SSE
  - GET  /health     → liveness probe

The SSE pattern matters here:
  Contract review with an LLM agent can take 10-30 seconds. Without streaming,
  the user stares at a spinner and bounces. With SSE, they see the agent's
  reasoning unfold in real time, which both feels faster AND builds trust
  (they can see what evidence the agent is actually looking at).
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from legallens.agents.loop import AgentLoopError, ContractReviewAgent
from legallens.models.domain import FollowUpQuestion, ReviewRequest

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hooks. We'll wire DB pools, Pinecone client, etc. here."""
    log.info("api.startup")
    # TODO: initialize DB pool, Pinecone client
    yield
    log.info("api.shutdown")
    # TODO: close DB pool, Pinecone client


app = FastAPI(
    title="LegalLens API",
    description="LLM-powered contract review assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/review")
async def review_contract(request: ReviewRequest) -> EventSourceResponse:
    """Start an agent-driven contract review. Streams events via SSE.

    Each event has shape: {"event": "<type>", "data": "<json>"}
    Types: thinking, tool_call, tool_result, final_answer, error
    """
    log.info("api.review.start", contract_id=request.contract_id)

    user_query = _build_review_prompt(request)
    agent = ContractReviewAgent()

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        try:
            async for event in agent.run(user_query):
                yield {
                    "event": event.type,
                    "data": event.model_dump_json(),
                }
        except AgentLoopError as e:
            yield {
                "event": "error",
                "data": f'{{"error": "{e}"}}',
            }

    return EventSourceResponse(event_stream())


@app.post("/followup")
async def follow_up(question: FollowUpQuestion) -> EventSourceResponse:
    """Ask a follow-up question. Same SSE pattern as /review."""
    log.info("api.followup", contract_id=question.contract_id)

    user_query = _build_followup_prompt(question)
    agent = ContractReviewAgent()

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        try:
            async for event in agent.run(user_query):
                yield {
                    "event": event.type,
                    "data": event.model_dump_json(),
                }
        except AgentLoopError as e:
            yield {
                "event": "error",
                "data": f'{{"error": "{e}"}}',
            }

    return EventSourceResponse(event_stream())


# --- Prompt builders ---

def _build_review_prompt(request: ReviewRequest) -> str:
    base = (
        f"Please review contract {request.contract_id}. "
        f"Identify the 5-10 clauses most worth a senior lawyer's attention, "
        f"and for each, provide a 1-2 sentence rationale."
    )
    if request.focus_categories:
        cats = ", ".join(c.value for c in request.focus_categories)
        base += f" Focus on these categories: {cats}."
    return base


def _build_followup_prompt(question: FollowUpQuestion) -> str:
    base = f"User question about contract {question.contract_id}: {question.question}"
    if question.clause_id:
        base += f" (scoped to clause {question.clause_id})"
    return base
