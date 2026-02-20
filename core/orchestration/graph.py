"""Main LangGraph architecture for the multi-agent system."""

from langgraph.graph import StateGraph, END
from core.orchestration.state import SessionState
from core.orchestration.nodes import (
    init_session,
    therapist_ask,
    client_respond,
    human_input_node,
    coverage_check,
    risk_check,
    retrieve_context,
    diagnostician_draft,
    evidence_audit,
    finalize_session,
    safe_exit,
)

def build_graph(checkpointer=None) -> StateGraph:
    """Builds the multi-agent orchestration graph.

    Flow:
    init -> therapist_ask -> risk_check -> [HUMAN or CLIENT_SIM] -> risk_check
        -> coverage_check
            |- (pending)  -> therapist_ask  [loop]
            |- (complete) -> retrieve_context -> diagnostician_draft
                           -> evidence_audit -> finalize

    RiskGate intercepts after every text generation step.
    """

    graph = StateGraph(SessionState)

    # --- Nodes ---
    graph.add_node("init_session", init_session)
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
    graph.add_edge("init_session", "therapist_ask")
    graph.add_edge("therapist_ask", "risk_check")

    # Risk check interception right after Therapist
    graph.add_conditional_edges("risk_check", _route_risk, {
        "auto": "client_respond",                # Direct to auto client if safe + auto
        "interactive": "human_input",            # Direct to human if safe + interactive
        "safe_diagnostician": "evidence_audit",   
        "safe_client": "coverage_check",          
        "safe_human": "coverage_check",
        "risky": "safe_exit",                     
    })

    graph.add_edge("client_respond", "risk_check")
    graph.add_edge("human_input", "risk_check")

    # Coverage check conditional routing
    graph.add_conditional_edges("coverage_check", _route_coverage, {
        "continue": "therapist_ask",          
        "complete": "retrieve_context",       
        "max_turns": "retrieve_context",      
    })

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
    if step == "therapist_ask":
        return _determine_client_mode(state) # Auto route
    elif step == "client_respond":
        return "safe_client"
    elif step == "human_input":
        return "safe_human"
    elif step == "diagnostician_draft":
        return "safe_diagnostician"
    return "safe_client"

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
