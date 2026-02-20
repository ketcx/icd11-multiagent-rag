"""Tests for bilingual prompt selectors."""

from __future__ import annotations

from core.agents.prompts import (
    THERAPIST_PROMPT,  # backward-compat alias
    get_auditor_prompt,
    get_client_prompt,
    get_diagnostician_prompt,
    get_therapist_prompt,
)


class TestPromptSelectors:
    def test_therapist_english(self) -> None:
        prompt = get_therapist_prompt("English")
        assert "empathetic" in prompt or "psychologist" in prompt

    def test_therapist_spanish(self) -> None:
        prompt = get_therapist_prompt("Español")
        assert "psicólogo" in prompt or "empático" in prompt

    def test_client_english(self) -> None:
        prompt = get_client_prompt("English")
        assert "patient" in prompt

    def test_client_spanish(self) -> None:
        prompt = get_client_prompt("Español")
        assert "paciente" in prompt

    def test_diagnostician_english_json_instruction(self) -> None:
        prompt = get_diagnostician_prompt("English")
        assert "JSON" in prompt

    def test_diagnostician_spanish_json_instruction(self) -> None:
        prompt = get_diagnostician_prompt("Español")
        assert "JSON" in prompt

    def test_auditor_english(self) -> None:
        prompt = get_auditor_prompt("English")
        assert len(prompt) > 0

    def test_auditor_spanish(self) -> None:
        prompt = get_auditor_prompt("Español")
        assert len(prompt) > 0

    def test_unknown_language_falls_back_to_english(self) -> None:
        # Any language other than 'Español' should return English
        prompt = get_therapist_prompt("Français")
        assert "psychologist" in prompt or "empathetic" in prompt

    def test_backward_compat_alias(self) -> None:
        """THERAPIST_PROMPT constant must still exist (backward compatibility)."""
        assert THERAPIST_PROMPT is not None
        assert len(THERAPIST_PROMPT) > 0
