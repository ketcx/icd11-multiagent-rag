"""Tests for EvidenceAuditorAgent."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from core.agents.auditor import EvidenceAuditorAgent
from core.agents.prompts import AUDITOR_PROMPT_EN


def _make_auditor(llm_response: str = "") -> EvidenceAuditorAgent:
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": llm_response}}]
    }
    return EvidenceAuditorAgent(
        llm=mock_llm,
        system_prompt=AUDITOR_PROMPT_EN,
        temperature=0.1,
        max_tokens=512,
    )


@pytest.fixture
def state_with_grounded_evidence() -> dict:
    return {
        "session_id": "test",
        "language": "English",
        "transcript": [
            {"role": "therapist", "content": "How are you sleeping?"},
            {"role": "client", "content": "I have been feeling very anxious and unable to sleep."},
        ],
        "retrieved_chunks": [
            {
                "content": "Generalised Anxiety Disorder 6B00 characterised by excessive anxiety worry",
                "metadata": {"code": "6B00"},
            }
        ],
        "hypotheses": [
            {
                "label": "Generalised Anxiety Disorder",
                "code": "6B00",
                "confidence": "HIGH",
                "evidence_for": ["anxious", "unable to sleep"],
                "evidence_against": [],
            }
        ],
    }


@pytest.fixture
def state_with_ungrounded_evidence() -> dict:
    return {
        "session_id": "test",
        "language": "English",
        "transcript": [
            {"role": "client", "content": "I feel tired."},
        ],
        "retrieved_chunks": [{"content": "Some ICD-11 context", "metadata": {}}],
        "hypotheses": [
            {
                "label": "Bipolar Disorder",
                "code": "6A60",
                "confidence": "LOW",
                "evidence_for": ["grandiosity", "racing thoughts", "decreased need for sleep"],
                "evidence_against": [],
            }
        ],
    }


class TestEvidenceAuditor:
    def test_audit_report_present(self, state_with_grounded_evidence: dict) -> None:
        auditor = _make_auditor()
        result = auditor.act(state_with_grounded_evidence)
        assert "audit_report" in result
        assert result["audit_report"] is not None

    def test_report_has_required_keys(self, state_with_grounded_evidence: dict) -> None:
        auditor = _make_auditor()
        result = auditor.act(state_with_grounded_evidence)
        report = result["audit_report"]
        for key in ("verified", "traceability_score", "total_claims", "grounded_claims", "issues"):
            assert key in report

    def test_grounded_evidence_verified(self, state_with_grounded_evidence: dict) -> None:
        auditor = _make_auditor()
        result = auditor.act(state_with_grounded_evidence)
        report = result["audit_report"]
        assert report["traceability_score"] > 0.0

    def test_ungrounded_evidence_produces_issues(
        self, state_with_ungrounded_evidence: dict
    ) -> None:
        auditor = _make_auditor()
        result = auditor.act(state_with_ungrounded_evidence)
        report = result["audit_report"]
        assert len(report["issues"]) > 0
        assert report["verified"] is False

    def test_traceability_score_between_0_and_1(self, state_with_grounded_evidence: dict) -> None:
        auditor = _make_auditor()
        result = auditor.act(state_with_grounded_evidence)
        score = result["audit_report"]["traceability_score"]
        assert 0.0 <= score <= 1.0

    def test_empty_hypotheses_is_fully_verified(self) -> None:
        auditor = _make_auditor()
        state = {
            "session_id": "test",
            "language": "English",
            "transcript": [],
            "retrieved_chunks": [],
            "hypotheses": [],
        }
        result = auditor.act(state)
        assert result["audit_report"]["verified"] is True
        assert result["audit_report"]["traceability_score"] == 1.0

    def test_no_llm_still_produces_report(self) -> None:
        auditor = EvidenceAuditorAgent(
            llm=None,
            system_prompt=AUDITOR_PROMPT_EN,
        )
        state = {
            "session_id": "test",
            "language": "English",
            "transcript": [{"role": "client", "content": "anxious"}],
            "retrieved_chunks": [],
            "hypotheses": [
                {
                    "label": "GAD",
                    "code": "6B00",
                    "confidence": "MEDIUM",
                    "evidence_for": ["anxious"],
                    "evidence_against": [],
                }
            ],
        }
        result = auditor.act(state)
        assert result["audit_report"] is not None
        assert result["audit_report"]["llm_commentary"] is None
