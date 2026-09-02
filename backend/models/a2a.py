"""A2A (Agent-to-Agent) protocol message models."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class A2AError(BaseModel):
    """Structured error within an A2A response."""

    error_type: str  # "inference_failure", "data_error", "timeout", "protocol_error"
    message: str
    originating_agent: str


class A2AMessage(BaseModel):
    """Task request sent from one agent to another via A2A protocol."""

    task_type: str  # "aero_eval", "structural_eval", "cost_eval", "surface_data"
    target_agent: str  # "aero", "structural", "cost"
    correlation_id: str
    payload: dict = Field(default_factory=dict)
    source_agent: str = "orchestrator"
    timestamp: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize to dictionary for wire transport."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> A2AMessage:
        """Deserialize from dictionary."""
        return cls.model_validate(data)


class A2AResponse(BaseModel):
    """Response returned from a specialist agent to the caller."""

    correlation_id: str
    source_agent: str
    status: str  # "success" or "error"
    payload: dict = Field(default_factory=dict)
    error: A2AError | None = None
    timestamp: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize to dictionary for wire transport."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> A2AResponse:
        """Deserialize from dictionary."""
        return cls.model_validate(data)
