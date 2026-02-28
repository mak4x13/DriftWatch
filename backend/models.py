"""Pydantic models for DriftWatch API contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class PipelineStep(BaseModel):
    """A single executable step in the user-defined pipeline."""

    step_id: str
    name: str
    instruction: str
    input_context: str


class PipelineRequest(BaseModel):
    """The full pipeline execution request."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    user_goal: str
    steps: list[PipelineStep]
    source_documents: list[str]
    auto_fix: bool = True


class SourceDocument(BaseModel):
    """A source document payload for explicit ingestion."""

    text: str
    run_id: str = Field(default_factory=lambda: str(uuid4()))


class AuditIssue(BaseModel):
    """A single detected problem in a pipeline step output."""

    type: Literal["INTENT_DRIFT", "HALLUCINATION", "CONTRADICTION"]
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    claim: str
    reason: str
    suggestion: str


class AuditResult(BaseModel):
    """The audit verdict and any applied correction for one step."""

    step_id: str
    verdict: Literal["PASS", "FLAG"]
    drift_score: float
    issues: list[AuditIssue]
    summary: str
    original_output: str
    final_output: str
    auto_fixed: bool = False


class AuditEvent(BaseModel):
    """A real-time audit stream event sent over SSE."""

    event_type: Literal[
        "STEP_START",
        "STEP_COMPLETE",
        "AUDIT_RESULT",
        "AUTOFIX",
        "PIPELINE_DONE",
        "ERROR",
    ]
    step_id: str
    data: dict[str, Any]
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PipelineResult(BaseModel):
    """The completed pipeline result and aggregate trust metrics."""

    run_id: str
    user_goal: str
    audit_results: list[AuditResult]
    final_output: str
    overall_drift_score: float
    total_hallucinations: int
    total_corrections: int
    pipeline_trustworthy: bool
