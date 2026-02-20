"""Tests for BaseAgent and common agent behaviour."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from core.agents.base import BaseAgent
from core.agents.prompts import THERAPIST_PROMPT_EN


# ---------------------------------------------------------------------------
# Concrete stub for testing the abstract base
# ---------------------------------------------------------------------------


class _ConcreteAgent(BaseAgent):
    """Minimal concrete implementation of BaseAgent for testing."""

    def act(self, state: dict) -> dict:
        response = self._generate([{"role": "user", "content": "Hello"}])
        state["last_response"] = response
        return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseAgentGenerate:
    def test_returns_llm_content(self) -> None:
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        agent = _ConcreteAgent(llm=mock_llm, system_prompt=THERAPIST_PROMPT_EN)
        result = agent._generate([{"role": "user", "content": "Hello"}])
        assert result == "Test response"

    def test_system_prompt_prepended(self) -> None:
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        agent = _ConcreteAgent(llm=mock_llm, system_prompt="MY_SYSTEM_PROMPT")
        agent._generate([{"role": "user", "content": "Hi"}])

        call_args = mock_llm.create_chat_completion.call_args
        messages_sent = call_args.kwargs.get("messages") or call_args.args[0]
        assert messages_sent[0]["role"] == "system"
        assert messages_sent[0]["content"] == "MY_SYSTEM_PROMPT"

    def test_returns_empty_string_on_none_llm(self) -> None:
        agent = _ConcreteAgent(llm=None, system_prompt=THERAPIST_PROMPT_EN)
        result = agent._generate([{"role": "user", "content": "Hi"}])
        assert result == ""

    def test_retries_on_transient_failure(self) -> None:
        mock_llm = MagicMock()
        # Fail once, then succeed
        mock_llm.create_chat_completion.side_effect = [
            RuntimeError("transient"),
            {"choices": [{"message": {"content": "recovered"}}]},
        ]
        agent = _ConcreteAgent(llm=mock_llm, system_prompt=THERAPIST_PROMPT_EN)
        result = agent._generate([{"role": "user", "content": "Hi"}])
        assert result == "recovered"
        assert mock_llm.create_chat_completion.call_count == 2

    def test_returns_empty_string_after_all_retries_fail(self) -> None:
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = RuntimeError("always fails")
        agent = _ConcreteAgent(llm=mock_llm, system_prompt=THERAPIST_PROMPT_EN)
        result = agent._generate([{"role": "user", "content": "Hi"}])
        assert result == ""

    def test_temperature_passed_to_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        agent = _ConcreteAgent(llm=mock_llm, system_prompt=THERAPIST_PROMPT_EN, temperature=0.42)
        agent._generate([{"role": "user", "content": "Hi"}])
        call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.42

    def test_max_tokens_passed_to_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        agent = _ConcreteAgent(llm=mock_llm, system_prompt=THERAPIST_PROMPT_EN, max_tokens=128)
        agent._generate([{"role": "user", "content": "Hi"}])
        call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
        assert call_kwargs["max_tokens"] == 128
