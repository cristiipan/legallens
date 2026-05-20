"""The core agent loop.

This module owns the conversation with Cohere Command R:
  1. Send user query + tool definitions to Cohere
  2. If Cohere returns tool_calls, execute them, send results back
  3. Loop until Cohere returns a final answer (no more tool_calls)
  4. Stream every step as AgentEvent for the SSE endpoint

The loop is intentionally simple. Production agents add things like:
  - max_iterations to prevent infinite loops
  - parallel tool execution
  - tool result caching
  - structured error recovery

We implement max_iterations here (basics matter), defer the rest until needed.
"""
from collections.abc import AsyncIterator
from typing import Any

import cohere
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from legallens.config import get_settings
from legallens.models.domain import AgentEvent
from legallens.prompts.tools import SYSTEM_PREAMBLE, TOOL_DEFINITIONS
from legallens.retrieval.tools import dispatch_tool

log = structlog.get_logger()

MAX_AGENT_ITERATIONS = 8  # safety net against infinite tool-call loops


class AgentLoopError(Exception):
    """Raised when the agent fails after retries or exceeds max iterations."""


class ContractReviewAgent:
    """Stateless agent. Each run() is an independent reasoning session."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = cohere.AsyncClient(api_key=settings.cohere_api_key)
        self.model = settings.cohere_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _call_cohere(
        self,
        message: str,
        chat_history: list[dict[str, Any]],
        tool_results: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Wrap Cohere API call with retry. Network blips are common; agent
        runs should not die because of a single transient failure."""
        return await self.client.chat(
            model=self.model,
            message=message,
            chat_history=chat_history,
            preamble=SYSTEM_PREAMBLE,
            tools=TOOL_DEFINITIONS,
            tool_results=tool_results,
        )

    async def run(self, user_query: str) -> AsyncIterator[AgentEvent]:
        """Run the agent loop, yielding events as they happen.

        Yields:
            AgentEvent: each thinking step, tool call, tool result, and the
                final answer. Consumers (typically the SSE endpoint) push
                these to the client.
        """
        chat_history: list[dict[str, Any]] = []
        current_message = user_query
        tool_results: list[dict[str, Any]] | None = None

        for iteration in range(MAX_AGENT_ITERATIONS):
            log.info("agent.iteration", n=iteration, has_tool_results=bool(tool_results))

            try:
                response = await self._call_cohere(
                    message=current_message,
                    chat_history=chat_history,
                    tool_results=tool_results,
                )
            except Exception as e:
                log.exception("agent.cohere_call_failed", iteration=iteration)
                yield AgentEvent(type="error", content=f"LLM call failed: {e}")
                raise AgentLoopError(str(e)) from e

            # Case 1: agent wants to call tools
            if response.tool_calls:
                yield AgentEvent(
                    type="thinking",
                    content=f"Calling {len(response.tool_calls)} tool(s)...",
                )

                tool_results = []
                for tool_call in response.tool_calls:
                    yield AgentEvent(
                        type="tool_call",
                        tool_name=tool_call.name,
                        content=f"Args: {tool_call.parameters}",
                    )

                    try:
                        result = await dispatch_tool(
                            name=tool_call.name,
                            parameters=tool_call.parameters,
                        )
                        yield AgentEvent(
                            type="tool_result",
                            tool_name=tool_call.name,
                            content=str(result)[:200],  # truncate for stream
                        )
                        tool_results.append({
                            "call": tool_call,
                            "outputs": [result] if isinstance(result, dict) else result,
                        })
                    except Exception as e:
                        log.exception("agent.tool_failed", tool=tool_call.name)
                        # Don't crash the whole loop — tell the agent the tool failed
                        # and let it decide how to recover.
                        tool_results.append({
                            "call": tool_call,
                            "outputs": [{"error": str(e)}],
                        })

                # Continue loop: feed tool results back to Cohere
                current_message = ""  # tool_results-only turn
                continue

            # Case 2: agent gave a final answer
            yield AgentEvent(type="final_answer", content=response.text)
            return

        # Hit max iterations without a final answer
        yield AgentEvent(
            type="error",
            content=f"Agent exceeded {MAX_AGENT_ITERATIONS} iterations without finishing.",
        )
        raise AgentLoopError("max iterations exceeded")
