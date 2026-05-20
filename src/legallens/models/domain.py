"""Domain models for LegalLens.

These are the data shapes that flow between layers (API <-> agent <-> retrieval).
Keeping them in one place makes the system easier to reason about.
"""
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ClauseCategory(str, Enum):
    """The 41 categories CUAD annotates. Listing the high-priority ones here;
    full list lives in scripts/ingest.py.
    """
    GOVERNING_LAW = "governing_law"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    INDEMNIFICATION = "indemnification"
    TERMINATION = "termination"
    NON_COMPETE = "non_compete"
    EXCLUSIVITY = "exclusivity"
    IP_OWNERSHIP = "ip_ownership"
    # ... 34 more in the real ingest
    OTHER = "other"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Clause(BaseModel):
    """A single extracted clause from a contract."""
    id: str
    contract_id: str
    category: ClauseCategory
    text: str = Field(..., description="The raw clause text from the contract")
    page: int | None = None
    risk_level: RiskLevel | None = None
    risk_rationale: str | None = Field(
        default=None,
        description="LLM-generated explanation of why this clause is rated at this risk level"
    )


class Contract(BaseModel):
    """Metadata for an ingested contract."""
    id: str
    filename: str
    contract_type: str | None = None  # e.g., "NDA", "Services Agreement"
    parties: list[str] = Field(default_factory=list)
    effective_date: datetime | None = None
    ingested_at: datetime


# --- API request/response models ---

class ReviewRequest(BaseModel):
    """Request to review a contract that's already been ingested."""
    contract_id: str
    focus_categories: list[ClauseCategory] | None = Field(
        default=None,
        description="If provided, only surface clauses in these categories"
    )


class AgentEvent(BaseModel):
    """A single event in the agent's reasoning stream (sent over SSE)."""
    type: Literal["thinking", "tool_call", "tool_result", "final_answer", "error"]
    content: str
    tool_name: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FollowUpQuestion(BaseModel):
    """A user follow-up about a specific clause or the contract overall."""
    contract_id: str
    question: str
    clause_id: str | None = Field(
        default=None,
        description="If the question is scoped to a specific clause"
    )
