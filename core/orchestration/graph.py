"""Main LangGraph architecture for the multi-agent system."""

from langgraph.graph import END, StateGraph

from core.orchestration.nodes import (
    client_respond,
    coverage_check,
    diagnostician_draft,
    evidence_audit,
    finalize_session,
    human_input_node,
    init_session,
    rapport_ask,
    rapport_coverage_check,
    retrieve_context,
    risk_check,
    safe_exit,
    therapist_ask,
)
from core.orchestration.state import SessionState


def build_graph(checkpointer=None) -> StateGraph:
    """Builds the multi-agent orchestration graph.

    Flow:
    init -> rapport_ask -> risk_check -> [HUMAN or CLIENT_SIM] -> risk_check
        -> rapport_coverage_check
            |- (continue_rapport) -> rapport_ask  [loop until rapport done]
            |- (to_clinical)      -> therapist_ask
        -> coverage_check
            |- (pending)  -> therapist_ask  [loop]
            |- (complete) -> retrieve_context -> diagnostician_draft
                           -> evidence_audit -> finalize

    RiskGate intercepts after every text generation step.
    """

    graph = StateGraph(SessionState)

    # --- Nodes ---
    graph.add_node("init_session", init_session)
    graph.add_node("rapport_ask", rapport_ask)
    graph.add_node("rapport_coverage_check", rapport_coverage_check)
    graph.add_node("therapist_ask", therapist_ask)
    graph.add_node("client_respond", client_respond)
    graph.add_node("human_input", human_input_node)
    graph.add_node("coverage_check", coverage_check)
    graph.add_node("risk_check", risk_check)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("diagnostician_draft", diagnostician_draft)
    graph.add_node("evidence_audit", evidence_audit)
    graph.add_node("finalize_session", finalize_session)
    graph.add_node("safe_exit", safe_exit)

    # --- Edges ---
    graph.set_entry_point("init_session")
    # Session opens with rapport phase, not clinical domains
    graph.add_edge("init_session", "rapport_ask")
    graph.add_edge("rapport_ask", "risk_check")
    graph.add_edge("therapist_ask", "risk_check")

    # Risk check: routes differ for rapport vs clinical phases
    graph.add_conditional_edges(
        "risk_check",
        _route_risk,
        {
            "auto": "client_respond",
            "interactive": "human_input",
            "rapport_check": "rapport_coverage_check",  # after rapport client/human turn
            "safe_diagnostician": "evidence_audit",
            "safe_client": "coverage_check",
            "safe_human": "coverage_check",
            "risky": "safe_exit",
        },
    )

    graph.add_edge("client_respond", "risk_check")
    graph.add_edge("human_input", "risk_check")

    # Rapport coverage: loop rapport or transition to clinical
    graph.add_conditional_edges(
        "rapport_coverage_check",
        _route_rapport,
        {
            "continue_rapport": "rapport_ask",
            "to_clinical": "therapist_ask",
        },
    )

    # Clinical coverage check conditional routing
    graph.add_conditional_edges(
        "coverage_check",
        _route_coverage,
        {
            "continue": "therapist_ask",
            "complete": "retrieve_context",
            "max_turns": "retrieve_context",
        },
    )

    graph.add_edge("retrieve_context", "diagnostician_draft")
    graph.add_edge("diagnostician_draft", "risk_check")
    graph.add_edge("evidence_audit", "finalize_session")
    graph.add_edge("finalize_session", END)
    graph.add_edge("safe_exit", END)

    # Compile the graph marking human_input as an interrupt point so Streamlit can inject
    return graph.compile(interrupt_before=["human_input"], checkpointer=checkpointer)


def _route_risk(state: SessionState) -> str:
    """Determines the next node based on risk_detected and current_step."""
    if state["risk_detected"]:
        return "risky"
    step = state["current_step"]
    rapport_done = state.get("rapport_complete", False)

    if step in ("therapist_ask", "rapport_ask"):
        # Both phases route to client/human the same way
        return _determine_client_mode(state)
    elif step == "client_respond":
        # If rapport is not yet done, go to rapport coverage check
        return "safe_client" if rapport_done else "rapport_check"
    elif step == "human_input":
        return "safe_human" if rapport_done else "rapport_check"
    elif step == "diagnostician_draft":
        return "safe_diagnostician"
    return "safe_client"


def _route_rapport(state: SessionState) -> str:
    """Routes after rapport_coverage_check: continue rapport or start clinical."""
    return "to_clinical" if state.get("rapport_complete", False) else "continue_rapport"


def _determine_client_mode(state: SessionState) -> str:
    """Checks if we are in interactive mode or synthetic simulation mode."""
    if state.get("interactive_mode", False):
        return "interactive"
    return "auto"


def _route_coverage(state: SessionState) -> str:
    """Determines whether to continue the interview or proceed to diagnosis."""
    if state["coverage_complete"]:
        return "complete"
    if state["turn_count"] >= state["max_turns"]:
        return "max_turns"
    return "continue"
