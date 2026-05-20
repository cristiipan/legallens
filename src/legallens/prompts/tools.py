"""Tool definitions exposed to the Cohere agent.

This is one of the most important files in the project. The agent's behavior
is shaped almost entirely by:
  1. The tool names and descriptions (Cohere uses them to decide when to call)
  2. The parameter schemas (must be JSON-schema-like dicts)
  3. The actual implementation in `dispatch()`

Each tool is intentionally small and single-purpose. Bigger tools = vaguer
agent behavior.
"""
from typing import Any

# Cohere uses a dict-based tool schema (similar to OpenAI's function calling).
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "extract_clauses",
        "description": (
            "Extract all clauses of a specific category from the contract. "
            "Use this when the user asks about a particular type of clause "
            "(e.g., 'show me the termination clauses') or when scanning "
            "the contract for known risk categories."
        ),
        "parameter_definitions": {
            "contract_id": {
                "description": "The ID of the contract to extract from",
                "type": "str",
                "required": True,
            },
            "category": {
                "description": (
                    "The clause category to extract. Must be one of: "
                    "governing_law, limitation_of_liability, indemnification, "
                    "termination, non_compete, exclusivity, ip_ownership, other"
                ),
                "type": "str",
                "required": True,
            },
        },
    },
    {
        "name": "search_similar_clauses",
        "description": (
            "Semantic search across the clause database to find clauses similar "
            "to a given query or example. Useful for finding precedents, "
            "comparing language across contracts, or finding unusual phrasings."
        ),
        "parameter_definitions": {
            "query": {
                "description": "Natural language query or example clause text",
                "type": "str",
                "required": True,
            },
            "top_k": {
                "description": "How many results to return (default 5)",
                "type": "int",
                "required": False,
            },
        },
    },
    {
        "name": "score_clause_risk",
        "description": (
            "Score a specific clause for risk level (low/medium/high) and "
            "generate a rationale. Use this once you've identified candidate "
            "clauses worth flagging to the user."
        ),
        "parameter_definitions": {
            "clause_id": {
                "description": "The ID of the clause to score",
                "type": "str",
                "required": True,
            },
        },
    },
]


SYSTEM_PREAMBLE = """You are LegalLens, a contract review assistant for legal teams.

Your job:
1. When given a contract, identify the 5-10 clauses most worth a senior \
lawyer's attention.
2. For each flagged clause, provide a short rationale (1-2 sentences) \
explaining why it deserves review.
3. When the user asks follow-up questions, ground your answers in the \
actual contract text by citing clause IDs.

Operating principles:
- Be specific. Vague answers like "the termination clause has some risk" \
are useless. Quote the language.
- When unsure, surface ambiguity rather than fabricate. It is much better \
to say "this clause is unusual, recommend human review" than to invent a \
confident-sounding analysis.
- You are a tool for lawyers, not a replacement for them. Your output is \
a starting point for review, not a legal opinion.

Use the provided tools to access the contract. Never fabricate clause text \
or IDs that you didn't get from a tool result."""
