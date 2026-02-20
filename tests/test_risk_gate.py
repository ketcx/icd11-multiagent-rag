"""Tests for RiskGate — bilingual keyword safety detection."""

from __future__ import annotations

import pytest

from core.safety.risk_gate import RiskGate


@pytest.fixture
def gate() -> RiskGate:
    return RiskGate()


class TestRiskGateSpanish:
    def test_suicid_detected(self, gate: RiskGate) -> None:
        is_risky, risk_type = gate.check("Estoy pensando en el suicidio")
        assert is_risky is True
        assert risk_type is not None

    def test_matarme_detected(self, gate: RiskGate) -> None:
        is_risky, _ = gate.check("Quiero matarme")
        assert is_risky is True

    def test_quitarme_la_vida_detected(self, gate: RiskGate) -> None:
        is_risky, _ = gate.check("He pensado en quitarme la vida")
        assert is_risky is True

    def test_autolesion_detected(self, gate: RiskGate) -> None:
        is_risky, _ = gate.check("Tengo pensamientos de autolesion")
        assert is_risky is True

    def test_no_quiero_vivir_detected(self, gate: RiskGate) -> None:
        is_risky, _ = gate.check("no quiero vivir más así")
        assert is_risky is True

    def test_cortarme_detected(self, gate: RiskGate) -> None:
        is_risky, _ = gate.check("Pensé en cortarme los brazos")
        assert is_risky is True

    def test_case_insensitive(self, gate: RiskGate) -> None:
        is_risky, _ = gate.check("SUICIDIO")
        assert is_risky is True

    def test_safe_text_not_flagged(self, gate: RiskGate) -> None:
        is_risky, risk_type = gate.check("Me siento un poco triste y cansado.")
        assert is_risky is False
        assert risk_type is None

    def test_empty_string_safe(self, gate: RiskGate) -> None:
        is_risky, risk_type = gate.check("")
        assert is_risky is False
        assert risk_type is None


class TestRiskGateEnglish:
    def test_kill_myself_detected(self, gate: RiskGate) -> None:
        is_risky, _ = gate.check("I want to kill myself")
        assert is_risky is True

    def test_suicid_english_detected(self, gate: RiskGate) -> None:
        is_risky, _ = gate.check("I've been having suicidal thoughts")
        assert is_risky is True

    def test_safe_english_text(self, gate: RiskGate) -> None:
        is_risky, _ = gate.check("I feel a bit down today.")
        assert is_risky is False


class TestRiskGateSafeResponse:
    def test_safe_response_not_empty(self, gate: RiskGate) -> None:
        response = gate.get_safe_response("Riesgo de Autolesión")
        assert len(response) > 0

    def test_safe_response_contains_emergency(self, gate: RiskGate) -> None:
        response = gate.get_safe_response("suicidio")
        # Should mention an emergency contact or number
        assert any(char.isdigit() for char in response)
