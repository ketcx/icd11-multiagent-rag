"""Tests for ClientAgent — profile injection and response variety."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from core.agents.client import ClientAgent, _format_profile
from core.agents.prompts import CLIENT_PROMPT_EN, CLIENT_PROMPT_ES


def _make_client(llm_response: str = "I feel anxious.") -> ClientAgent:
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": llm_response}}]
    }
    return ClientAgent(llm=mock_llm, system_prompt=CLIENT_PROMPT_EN)


SAMPLE_PROFILE = {
    "profile_id": "test_anxiety",
    "demographics": {"name": "Alex", "age": 34, "gender": "non-binary"},
    "presenting_complaints": ["excessive worry", "insomnia", "restlessness"],
    "history": "Work-related stress for 6 months.",
    "language": "English",
}

SAMPLE_TRANSCRIPT = [
    {"role": "therapist", "content": "How has your sleep been lately?", "domain": "sleep", "turn_id": 0},
]


class TestClientAgentProfileInjection:
    def test_profile_name_appears_in_messages(self) -> None:
        agent = _make_client()
        messages = agent._build_messages(SAMPLE_TRANSCRIPT, SAMPLE_PROFILE, "English")
        # The profile context should be the first message
        assert any("Alex" in m["content"] for m in messages)

    def test_presenting_complaints_appear_in_messages(self) -> None:
        agent = _make_client()
        messages = agent._build_messages(SAMPLE_TRANSCRIPT, SAMPLE_PROFILE, "English")
        full_text = " ".join(m["content"] for m in messages)
        assert "worry" in full_text or "insomnia" in full_text

    def test_history_appears_in_messages(self) -> None:
        agent = _make_client()
        messages = agent._build_messages(SAMPLE_TRANSCRIPT, SAMPLE_PROFILE, "English")
        full_text = " ".join(m["content"] for m in messages)
        assert "stress" in full_text or "months" in full_text

    def test_empty_profile_does_not_crash(self) -> None:
        agent = _make_client()
        messages = agent._build_messages(SAMPLE_TRANSCRIPT, {}, "English")
        assert isinstance(messages, list)
        assert len(messages) >= 1

    def test_transcript_history_included(self) -> None:
        agent = _make_client()
        messages = agent._build_messages(SAMPLE_TRANSCRIPT, SAMPLE_PROFILE, "English")
        full_text = " ".join(m["content"] for m in messages)
        assert "sleep" in full_text

    def test_final_instruction_is_last_message(self) -> None:
        agent = _make_client()
        messages = agent._build_messages(SAMPLE_TRANSCRIPT, SAMPLE_PROFILE, "English")
        last = messages[-1]
        assert last["role"] == "user"
        assert "respond" in last["content"].lower() or "reply" in last["content"].lower()

    def test_profile_context_is_first_message(self) -> None:
        agent = _make_client()
        messages = agent._build_messages(SAMPLE_TRANSCRIPT, SAMPLE_PROFILE, "English")
        # With a non-empty profile, the first message should carry profile context
        first_user = next(m for m in messages if m["role"] == "user")
        assert "Alex" in first_user["content"]

    def test_spanish_instruction_in_spanish_mode(self) -> None:
        agent = _make_client()
        messages = agent._build_messages(SAMPLE_TRANSCRIPT, SAMPLE_PROFILE, "Español")
        last = messages[-1]
        # Should be in Spanish
        assert "responde" in last["content"].lower() or "español" in last["content"].lower()

    def test_act_appends_to_transcript(self) -> None:
        agent = _make_client("I feel very anxious today.")
        state = {
            "transcript": list(SAMPLE_TRANSCRIPT),
            "client_profile": SAMPLE_PROFILE,
            "language": "English",
        }
        result = agent.act(state)
        assert len(result["transcript"]) == 2
        assert result["transcript"][-1]["role"] == "client"
        assert result["transcript"][-1]["content"] == "I feel very anxious today."


class TestFormatProfile:
    def test_english_format_contains_name(self) -> None:
        text = _format_profile(SAMPLE_PROFILE, "English")
        assert "Alex" in text

    def test_english_format_contains_age(self) -> None:
        text = _format_profile(SAMPLE_PROFILE, "English")
        assert "34" in text

    def test_english_format_contains_complaints(self) -> None:
        text = _format_profile(SAMPLE_PROFILE, "English")
        assert "worry" in text or "insomnia" in text

    def test_spanish_format_contains_name(self) -> None:
        text = _format_profile(SAMPLE_PROFILE, "Español")
        assert "Alex" in text

    def test_spanish_format_uses_spanish_labels(self) -> None:
        text = _format_profile(SAMPLE_PROFILE, "Español")
        assert "Eres" in text or "Motivo" in text or "Historial" in text

    def test_empty_profile_returns_string(self) -> None:
        text = _format_profile({}, "English")
        assert isinstance(text, str)


class TestMockResponseVariety:
    """Validates that mock fallback responses vary by domain."""

    def test_mock_responses_differ_across_domains(self) -> None:
        from core.orchestration.nodes import _mock_client_response
        import random
        random.seed(0)
        domains = ["mood", "anxiety", "sleep", "eating", "trauma", "cognition"]
        responses = {d: _mock_client_response(d, "Español") for d in domains}
        # All responses should be different (they target different domains)
        assert len(set(responses.values())) == len(domains)

    def test_mock_therapist_questions_differ_across_domains(self) -> None:
        from core.orchestration.nodes import _mock_therapist_question
        import random
        random.seed(0)
        domains = ["mood", "anxiety", "sleep", "eating", "trauma", "cognition"]
        questions = {d: _mock_therapist_question(d, "English") for d in domains}
        assert len(set(questions.values())) == len(domains)

    def test_same_domain_can_return_different_responses(self) -> None:
        """Multiple calls for the same domain may return different options."""
        from core.orchestration.nodes import _mock_client_response
        # Collect all unique responses for 'mood' over many calls
        responses = {_mock_client_response("mood", "Español") for _ in range(50)}
        # The bank has 3 options for mood — at least 2 should appear in 50 draws
        assert len(responses) >= 2
