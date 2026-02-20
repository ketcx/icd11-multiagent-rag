"""Pydantic models for diagnostic output."""

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ConfidenceBand(str, Enum):
    LOW = "LOW"  # Was "BAJA"
    MEDIUM = "MEDIUM"  # Was "MEDIA"
    HIGH = "HIGH"  # Was "ALTA"


class EvidenceItem(BaseModel):
    source: str = Field(..., description="'transcript' or 'rag'")
    turn_id: int | None = Field(None, description="Turn ID if source is transcript")
    page: int | None = Field(None, description="PDF page if source is rag")
    section: str | None = Field(None, description="ICD-11 section")
    code: str | None = Field(None, description="ICD-11 code")
    text: str = Field(..., description="Cited text serving as evidence")


class Hypothesis(BaseModel):
    label: str = Field(..., description="Name of the disorder/condition")
    code: str | None = Field(None, description="ICD-11 code")
    confidence: ConfidenceBand
    evidence_for: list[EvidenceItem] = Field(default_factory=list)
    evidence_against: list[EvidenceItem] = Field(default_factory=list)


class DiagnosisOutput(BaseModel):
    """Complete output of an educational diagnostic session."""

    session_id: str
    timestamp: datetime
    transcript: list[dict]
    coverage: dict = Field(..., description="Covered domains and confidence levels")
    hypotheses: list[Hypothesis]
    audit_report: dict | None = None
    limitations: list[str] = Field(default_factory=list)
    next_steps_educational: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict, description="Model, temperature, seed, etc info")
