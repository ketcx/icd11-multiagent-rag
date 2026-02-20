"""Tests for DiagnosticianAgent JSON parsing logic."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from core.agents.diagnostician import DiagnosticianAgent
from core.agents.prompts import DIAGNOSTICIAN_PROMPT_EN


def _make_agent(llm_response: str) -> DiagnosticianAgent:
    """Returns a DiagnosticianAgent whose LLM always returns *llm_response*."""
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": llm_response}}]
    }
    return DiagnosticianAgent(
        llm=mock_llm,
        system_prompt=DIAGNOSTICIAN_PROMPT_EN,
        temperature=0.3,
        max_tokens=1024,
    )


@pytest.fixture
def minimal_state() -> dict:
    return {
        "session_id": "test",
        "transcript": [
            {"role": "therapist", "content": "How have you been feeling?"},
            {"role": "client", "content": "Very anxious and unable to sleep."},
        ],
        "retrieved_chunks": [
            {
                "content": "Generalised Anxiety Disorder - ICD-11 6B00: characterised by excessive anxiety.",
                "metadata": {"code": "6B00"},
            }
        ],
        "language": "English",
    }


class TestDiagnosticianJSONParsing:
    """Exercises the JSON extraction logic in DiagnosticianAgent.act()."""

    def test_plain_json_array(self, minimal_state: dict) -> None:
        raw = json.dumps(
            [
                {
                    "label": "GAD",
                    "code": "6B00",
                    "confidence": "HIGH",
                    "evidence_for": ["anxious"],
                    "evidence_against": [],
                }
            ]
        )
        agent = _make_agent(raw)
        result = agent.act(dict(minimal_state))
        assert isinstance(result["hypotheses"], list)
        assert result["hypotheses"][0]["code"] == "6B00"

    def test_json_in_code_fence(self, minimal_state: dict) -> None:
        raw = '```json\n[{"label": "GAD", "code": "6B00", "confidence": "MEDIUM", "evidence_for": [], "evidence_against": []}]\n```'
        agent = _make_agent(raw)
        result = agent.act(dict(minimal_state))
        assert result["hypotheses"][0]["confidence"] == "MEDIUM"

    def test_json_in_plain_code_fence(self, minimal_state: dict) -> None:
        raw = '```\n[{"label": "GAD", "code": "6B00", "confidence": "LOW", "evidence_for": [], "evidence_against": []}]\n```'
        agent = _make_agent(raw)
        result = agent.act(dict(minimal_state))
        assert result["hypotheses"][0]["confidence"] == "LOW"

    def test_malformed_json_returns_error_hypothesis(self, minimal_state: dict) -> None:
        agent = _make_agent("This is not valid JSON at all!")
        result = agent.act(dict(minimal_state))
        hypotheses = result["hypotheses"]
        assert len(hypotheses) == 1
        # Should gracefully degrade to an error entry
        assert hypotheses[0]["confidence"] in ("LOW", "BAJA")

    def test_trailing_comma_in_array_is_handled(self, minimal_state: dict) -> None:
        """LLMs frequently emit trailing commas; the parser must tolerate them."""
        raw = (
            '[{"label": "GAD", "code": "6B00", "confidence": "HIGH",'
            ' "evidence_for": ["worry",], "evidence_against": []},]'
        )
        agent = _make_agent(raw)
        result = agent.act(dict(minimal_state))
        assert isinstance(result["hypotheses"], list)
        assert result["hypotheses"][0]["code"] == "6B00"

    def test_trailing_comma_in_nested_object(self, minimal_state: dict) -> None:
        raw = (
            '[{"label": "GAD", "code": "6B00", "confidence": "MEDIUM",'
            ' "evidence_for": ["anxious",], "evidence_against": ["none",],}]'
        )
        agent = _make_agent(raw)
        result = agent.act(dict(minimal_state))
        assert result["hypotheses"][0]["code"] == "6B00"

    def test_single_object_wrapped_in_list(self, minimal_state: dict) -> None:
        raw = json.dumps(
            {
                "label": "GAD",
                "code": "6B00",
                "confidence": "HIGH",
                "evidence_for": [],
                "evidence_against": [],
            }
        )
        agent = _make_agent(raw)
        result = agent.act(dict(minimal_state))
        assert isinstance(result["hypotheses"], list)
        assert len(result["hypotheses"]) == 1

    def test_language_spanish(self, minimal_state: dict) -> None:
        state = dict(minimal_state)
        state["language"] = "Español"
        raw = json.dumps(
            [
                {
                    "label": "TAG",
                    "code": "6B00",
                    "confidence": "ALTA",
                    "evidence_for": ["ansiedad"],
                    "evidence_against": [],
                }
            ]
        )
        agent = _make_agent(raw)
        result = agent.act(state)
        assert result["hypotheses"][0]["label"] == "TAG"
