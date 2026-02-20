"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_profile() -> dict:
    """Returns a minimal synthetic client profile."""
    return {
        "profile_id": "test_001",
        "demographics": {"name": "Test User", "age": 30, "gender": "non-binary"},
        "presenting_complaints": ["anxiety", "insomnia"],
        "history": "Work-related stress for 3 months.",
        "language": "Español",
    }


@pytest.fixture
def base_session_state(sample_profile: dict) -> dict:
    """Returns a SessionState-compatible dict that mirrors the post-init_session state.

    ``domains_pending`` is explicitly populated so that ``coverage_check`` and
    other nodes behave as they would during a real session.
    """
    from core.agents.therapist import TherapistAgent

    return {
        "session_id": "test_session_001",
        "client_profile": sample_profile,
        "transcript": [],
        "messages": [],
        "domains_covered": [],
        "domains_pending": list(TherapistAgent.DOMAINS),  # all 11 domains pending
        "coverage_complete": False,
        "retrieved_chunks": [],
        "query_history": [],
        "hypotheses": [],
        "audit_report": None,
        "risk_detected": False,
        "risk_type": None,
        "current_step": "init",
        "turn_count": 0,
        "max_turns": 40,
        "finalized": False,
        "interactive_mode": False,
        "language": "Español",
    }
