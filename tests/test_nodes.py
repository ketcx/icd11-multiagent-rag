"""Tests for LangGraph node functions."""

from __future__ import annotations

import pytest
from core.orchestration.nodes import coverage_check, risk_check, init_session


class TestInitSession:
    def test_initialises_pending_domains(self, base_session_state: dict) -> None:
        result = init_session(base_session_state)
        assert "domains_pending" in result
        assert len(result["domains_pending"]) > 0

    def test_coverage_starts_false(self, base_session_state: dict) -> None:
        result = init_session(base_session_state)
        assert result["coverage_complete"] is False

    def test_risk_starts_false(self, base_session_state: dict) -> None:
        result = init_session(base_session_state)
        assert result["risk_detected"] is False

    def test_transcript_initialised_empty(self, base_session_state: dict) -> None:
        result = init_session(base_session_state)
        assert result["transcript"] == []

    def test_domains_shuffled_differently_across_sessions(self, base_session_state: dict) -> None:
        """Running init_session multiple times should produce different domain orders."""
        orders = set()
        for _ in range(20):
            state = dict(base_session_state)
            state["domains_pending"] = []  # force fresh shuffle each time
            result = init_session(state)
            orders.add(tuple(result["domains_pending"]))
        # With 11 domains, the chance of all 20 runs being identical is astronomically small
        assert len(orders) > 1, "Domain order should vary across sessions"

    def test_all_domains_present_after_shuffle(self, base_session_state: dict) -> None:
        from core.agents.therapist import TherapistAgent

        state = dict(base_session_state)
        state["domains_pending"] = []
        result = init_session(state)
        assert set(result["domains_pending"]) == set(TherapistAgent.DOMAINS)

    def test_existing_domains_pending_not_reshuffled(self, base_session_state: dict) -> None:
        """If domains_pending is already set (resume), do not reshuffle."""
        original_order = ["sleep", "mood", "anxiety"]
        state = dict(base_session_state)
        state["domains_pending"] = original_order
        result = init_session(state)
        assert result["domains_pending"] == original_order


class TestCoverageCheck:
    def test_incomplete_when_no_domains_covered(self, base_session_state: dict) -> None:
        result = coverage_check(base_session_state)
        assert result["coverage_complete"] is False
        assert len(result["domains_pending"]) > 0

    def test_complete_when_all_domains_in_transcript(self, base_session_state: dict) -> None:
        from core.agents.therapist import TherapistAgent

        state = dict(base_session_state)
        state["transcript"] = [
            {"role": "therapist", "content": "Q", "domain": d, "turn_id": i}
            for i, d in enumerate(TherapistAgent.DOMAINS)
        ]
        result = coverage_check(state)
        assert result["coverage_complete"] is True
        assert result["domains_pending"] == []

    def test_complete_when_max_turns_reached(self, base_session_state: dict) -> None:
        state = dict(base_session_state)
        state["turn_count"] = 40
        state["max_turns"] = 40
        result = coverage_check(state)
        assert result["coverage_complete"] is True

    def test_domains_derived_from_transcript(self, base_session_state: dict) -> None:
        state = dict(base_session_state)
        state["transcript"] = [
            {"role": "therapist", "content": "Q", "domain": "mood", "turn_id": 0},
            {"role": "therapist", "content": "Q", "domain": "anxiety", "turn_id": 1},
        ]
        result = coverage_check(state)
        assert "mood" in result["domains_covered"]
        assert "anxiety" in result["domains_covered"]


class TestRiskCheck:
    def test_safe_text_returns_no_risk(self, base_session_state: dict) -> None:
        state = dict(base_session_state)
        state["transcript"] = [
            {"role": "client", "content": "Me siento un poco cansado hoy.", "turn_id": 0}
        ]
        result = risk_check(state)
        assert result["risk_detected"] is False
        assert result["risk_type"] is None

    def test_risky_spanish_text_detected(self, base_session_state: dict) -> None:
        state = dict(base_session_state)
        state["transcript"] = [
            {"role": "client", "content": "He pensado en suicidarme.", "turn_id": 0}
        ]
        result = risk_check(state)
        assert result["risk_detected"] is True
        assert result["risk_type"] is not None

    def test_risky_english_text_detected(self, base_session_state: dict) -> None:
        state = dict(base_session_state)
        state["transcript"] = [
            {"role": "client", "content": "I want to kill myself.", "turn_id": 0}
        ]
        result = risk_check(state)
        assert result["risk_detected"] is True

    def test_empty_transcript_is_safe(self, base_session_state: dict) -> None:
        state = dict(base_session_state)
        state["transcript"] = []
        result = risk_check(state)
        assert result["risk_detected"] is False

    def test_only_last_turn_checked(self, base_session_state: dict) -> None:
        """Historical risky turns should not re-trigger; only the latest matters."""
        state = dict(base_session_state)
        state["transcript"] = [
            {"role": "client", "content": "He pensado en suicidarme.", "turn_id": 0},
            {"role": "client", "content": "Gracias por la ayuda.", "turn_id": 1},
        ]
        result = risk_check(state)
        # Last turn is safe
        assert result["risk_detected"] is False
