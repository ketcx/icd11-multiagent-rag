"""Session management endpoints.

Lifecycle
---------
POST /sessions           → create + init, returns session_id
POST /sessions/{id}/turn → advance one graph turn
GET  /sessions/{id}      → retrieve current state snapshot
GET  /sessions/{id}/transcript → retrieve transcript only
POST /sessions/{id}/finalize   → force diagnosis + finalise
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from apps.api.state import app_state

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    client_profile: dict
    language: str = "Español"
    interactive_mode: bool = False
    max_turns: int = 40


class TurnRequest(BaseModel):
    human_input: str | None = None  # Only required in interactive_mode


class SessionSummary(BaseModel):
    session_id: str
    current_step: str
    turn_count: int
    coverage_complete: bool
    risk_detected: bool
    finalized: bool
    hypotheses_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create and initialise a new session",
)
def create_session(body: CreateSessionRequest) -> dict:
    """Creates a new session from a client profile and returns the session ID.

    The session is initialised but the graph has not yet advanced — call
    ``POST /sessions/{id}/turn`` to begin the interview.
    """
    from core.orchestration.nodes import init_session

    session_id = (
        f"sess_{datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )

    initial_state: dict = {
        "session_id": session_id,
        "client_profile": body.client_profile,
        "transcript": [],
        "messages": [],
        "domains_covered": [],
        "domains_pending": [],
        "coverage_complete": False,
        "retrieved_chunks": [],
        "query_history": [],
        "hypotheses": [],
        "audit_report": None,
        "risk_detected": False,
        "risk_type": None,
        "current_step": "created",
        "turn_count": 0,
        "max_turns": body.max_turns,
        "finalized": False,
        "interactive_mode": body.interactive_mode,
        "language": body.language,
    }

    # Run init_session node to populate domains
    initial_state.update(init_session(initial_state))
    app_state.sessions[session_id] = initial_state

    return {"session_id": session_id, "domains_pending": initial_state["domains_pending"]}


@router.post("/{session_id}/turn", summary="Advance the session by one graph turn")
def execute_turn(session_id: str, body: TurnRequest) -> dict:
    """Runs the next therapist→client cycle.

    In **interactive mode**, ``human_input`` must be provided; it is injected
    as the client's response before the graph continues.  In **auto mode**,
    the ClientAgent generates the response.

    Returns:
        The latest therapist question, the client response, current step, and
        whether the session has been finalised or risk was detected.
    """
    state = _get_session_or_404(session_id)

    if state["finalized"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already finalised. Retrieve results via GET /sessions/{id}.",
        )

    if state["risk_detected"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session was halted by the safety gate.",
        )

    from core.orchestration.nodes import (
        client_respond,
        coverage_check,
        risk_check,
        therapist_ask,
    )

    # Therapist asks
    state.update(therapist_ask(state))

    # Risk check after therapist
    state.update(risk_check(state))
    if state["risk_detected"]:
        app_state.sessions[session_id] = state
        from core.safety.risk_gate import RiskGate

        gate = RiskGate()
        return {
            "risk_detected": True,
            "safe_response": gate.get_safe_response(state.get("risk_type", "")),
        }

    # Client responds (human or auto)
    if state["interactive_mode"]:
        if not body.human_input:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="human_input is required in interactive_mode.",
            )
        state["transcript"].append(
            {
                "role": "client",
                "content": body.human_input,
                "turn_id": len(state["transcript"]),
            }
        )
        state["turn_count"] = state.get("turn_count", 0) + 1
    else:
        state.update(client_respond(state))

    # Risk check after client
    state.update(risk_check(state))
    if state["risk_detected"]:
        app_state.sessions[session_id] = state
        from core.safety.risk_gate import RiskGate

        gate = RiskGate()
        return {
            "risk_detected": True,
            "safe_response": gate.get_safe_response(state.get("risk_type", "")),
        }

    # Update coverage
    state.update(coverage_check(state))
    app_state.sessions[session_id] = state

    latest_turns = (
        state["transcript"][-2:] if len(state["transcript"]) >= 2 else state["transcript"]
    )
    return {
        "turn_count": state["turn_count"],
        "coverage_complete": state["coverage_complete"],
        "risk_detected": False,
        "finalized": state["finalized"],
        "latest_turns": latest_turns,
    }


@router.post("/{session_id}/finalize", summary="Run diagnosis and finalise the session")
def finalize_session_endpoint(session_id: str) -> dict:
    """Triggers RAG retrieval, diagnosis, and evidence audit.

    This endpoint is called automatically after coverage is complete, or can
    be called manually to force finalisation before all domains are covered.
    """
    state = _get_session_or_404(session_id)

    if state["finalized"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already finalised.",
        )

    from core.orchestration.nodes import (
        diagnostician_draft,
        evidence_audit,
        retrieve_context,
    )

    state.update(retrieve_context(state))
    state.update(diagnostician_draft(state))
    state.update(evidence_audit(state))
    state["finalized"] = True
    state["current_step"] = "finalized"

    app_state.sessions[session_id] = state

    return {
        "session_id": session_id,
        "finalized": True,
        "hypotheses": state.get("hypotheses", []),
        "audit_report": state.get("audit_report"),
    }


@router.get("/{session_id}", summary="Retrieve the full session state")
def get_session(session_id: str) -> dict:
    """Returns the complete session state including transcript and hypotheses."""
    state = _get_session_or_404(session_id)
    return {
        "session_id": state["session_id"],
        "current_step": state["current_step"],
        "turn_count": state["turn_count"],
        "coverage_complete": state["coverage_complete"],
        "domains_covered": state["domains_covered"],
        "domains_pending": state["domains_pending"],
        "risk_detected": state["risk_detected"],
        "finalized": state["finalized"],
        "hypotheses": state.get("hypotheses", []),
        "audit_report": state.get("audit_report"),
        "transcript": state.get("transcript", []),
    }


@router.get("/{session_id}/transcript", summary="Retrieve the session transcript only")
def get_transcript(session_id: str) -> dict:
    """Returns only the conversation transcript (lighter payload)."""
    state = _get_session_or_404(session_id)
    return {
        "session_id": session_id,
        "turn_count": state["turn_count"],
        "transcript": state.get("transcript", []),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_session_or_404(session_id: str) -> dict:
    state = app_state.sessions.get(session_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return state
