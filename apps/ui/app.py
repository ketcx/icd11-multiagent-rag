"""Streamlit UI application for ICD-11 Multi-Agent RAG."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Generator
from pathlib import Path

import streamlit as st
import yaml

_logger = logging.getLogger(__name__)

from core.orchestration.graph import build_graph
from core.orchestration.nodes import AGENTS

# ---------------------------------------------------------------------------
# Internationalisation (i18n)
# ---------------------------------------------------------------------------

_STRINGS: dict[str, dict[str, str]] = {
    # Sidebar
    "sidebar_header": {"en": "⚙️ Configuration", "es": "⚙️ Configuración"},
    "llm_loaded": {"en": "✅ Model loaded", "es": "✅ Modelo cargado"},
    "llm_mock": {"en": "⚠️ No model (mock mode)", "es": "⚠️ Sin modelo (modo mock)"},
    "language_label": {"en": "Language / Idioma", "es": "Idioma / Language"},
    "mode_label": {"en": "Patient Mode", "es": "Modo de Paciente"},
    "mode_auto": {"en": "Auto (Simulated Profile)", "es": "Auto (Perfil Simulado)"},
    "mode_interactive": {"en": "Interactive (Human)", "es": "Interactivo (Humano)"},
    "max_turns_caption": {"en": "max turns", "es": "turnos máx."},
    "domains_caption": {"en": "domains", "es": "dominios"},
    "new_session_btn": {"en": "🔄 New Session", "es": "🔄 Nueva Sesión"},
    # Page title / header
    "page_title": {"en": "ICD-11 Multi-Agent RAG", "es": "ICD-11 Multi-Agent RAG"},
    "page_subtitle": {
        "en": "Educational clinical interview simulator",
        "es": "Simulador educativo de entrevista clínica",
    },
    "disclaimer": {
        "en": (
            "⚠️ **NOTICE:** This is an educational system only. "
            "It does **not** provide real clinical diagnosis or medical recommendations. "
            "Generated hypotheses are simulations for research purposes."
        ),
        "es": (
            "⚠️ **AVISO:** Este es un sistema exclusivamente educativo. "
            "**No** proporciona diagnóstico clínico real. "
            "Las hipótesis generadas son simulaciones para fines de investigación."
        ),
    },
    # Session control
    "btn_start_interactive": {
        "en": "▶️ Start Interactive Interview",
        "es": "▶️ Iniciar Entrevista Interactiva",
    },
    "btn_start_auto": {
        "en": "▶️ Start Automatic Simulation",
        "es": "▶️ Iniciar Simulación Automática",
    },
    "spinner_init": {"en": "Initialising session…", "es": "Inicializando sesión…"},
    "spinner_thinking": {"en": "Therapist is thinking…", "es": "El terapeuta está pensando…"},
    "spinner_simulating": {"en": "Simulating next turn…", "es": "Simulando siguiente turno…"},
    "chat_placeholder": {"en": "Type your response here…", "es": "Escribe tu respuesta aquí…"},
    # Progress info
    "progress_info": {
        "en": "Turn {turn}/{max} · Pending domains: {pending} · Covered: {covered}",
        "es": "Turno {turn}/{max} · Dominios pendientes: {pending} · Cubiertos: {covered}",
    },
    # Results
    "session_complete": {"en": "✅ Session complete", "es": "✅ Sesión completada"},
    "risk_terminated": {
        "en": "Session terminated by safety gate.",
        "es": "Sesión terminada por la puerta de seguridad.",
    },
    "hypotheses_header": {"en": "Diagnostic Hypotheses", "es": "Hipótesis Diagnósticas"},
    "no_hypotheses": {"en": "No hypotheses generated.", "es": "Sin hipótesis generadas."},
    "confidence_label": {"en": "Confidence", "es": "Confianza"},
    "evidence_for_label": {"en": "Supporting evidence", "es": "Evidencia a favor"},
    "evidence_against_label": {"en": "Contradicting evidence", "es": "Evidencia en contra"},
    "audit_header": {"en": "Evidence Audit", "es": "Auditoría de Evidencia"},
    "traceability_metric": {"en": "Traceability score", "es": "Puntuación de trazabilidad"},
    "issues_warning": {
        "en": "{n} claim(s) without direct grounding",
        "es": "{n} afirmación(es) sin respaldo directo",
    },
    "no_issues": {"en": "All claims traceable", "es": "Todas las afirmaciones trazables"},
    "auditor_comment": {"en": "Auditor commentary", "es": "Comentario del auditor"},
    "full_transcript": {"en": "Full Transcript", "es": "Transcripción Completa"},
    "view_transcript": {"en": "View transcript", "es": "Ver transcripción"},
    "domain_caption": {"en": "domain", "es": "dominio"},
    # Model loading spinner (must be a static string for @st.cache_resource)
    "spinner_loading_model": {"en": "Loading LLM into memory…", "es": "Cargando modelo LLM…"},
}


def t(key: str, lang: str | None = None, **kwargs: object) -> str:
    """Returns the UI string for *key* in the current session language.

    Args:
        key:    Translation key defined in ``_STRINGS``.
        lang:   Language override ("en"/"es"). Uses session language when omitted.
        **kwargs: Format arguments forwarded to ``str.format``.
    """
    if lang is None:
        raw = st.session_state.get("session_language", "English")
        lang = "es" if raw == "Español" else "en"
    text = _STRINGS.get(key, {}).get(lang, _STRINGS.get(key, {}).get("en", key))
    return text.format(**kwargs) if kwargs else text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "configs" / "app.yaml"
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Agent / model loading (with graceful mock fallback)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading LLM into memory — this may take a moment…")
def load_agents() -> tuple[dict, str, bool]:
    """Loads the LLM and instantiates all agents.

    Returns:
        Tuple of (agents dict, status message, llm_available flag).
        Falls back to mock mode (llm=None) when the GGUF is not cached locally.
    """
    cfg = _load_config()
    from core.agents import create_llm
    from core.agents.auditor import EvidenceAuditorAgent
    from core.agents.client import ClientAgent
    from core.agents.diagnostician import DiagnosticianAgent
    from core.agents.prompts import (
        get_auditor_prompt,
        get_client_prompt,
        get_diagnostician_prompt,
        get_therapist_prompt,
    )
    from core.agents.therapist import TherapistAgent
    from core.retrieval import init_rag_pipeline

    llm = None
    llm_available = False

    try:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=cfg["llm"]["model_name"],
            filename=cfg["llm"]["model_file"],
            local_files_only=True,  # only use cached copy; never trigger a download here
        )
        llm = create_llm(
            model_path,
            n_ctx=cfg["llm"]["n_ctx"],
            n_gpu_layers=cfg["llm"]["n_gpu_layers"],
            chat_format=cfg["llm"]["chat_format"],
        )
        llm_available = True
    except Exception:
        pass  # mock mode — llm stays None

    # Agents are initialised with a neutral language; the system prompt is
    # re-selected per-call via get_<agent>_prompt(state["language"]).
    lang = "English"
    agents = {
        "therapist": TherapistAgent(
            llm,
            system_prompt=get_therapist_prompt(lang),
            temperature=cfg["agents"]["therapist"]["temperature"],
            max_tokens=cfg["agents"]["therapist"]["max_tokens"],
        ),
        "client": ClientAgent(
            llm,
            system_prompt=get_client_prompt(lang),
            temperature=cfg["agents"]["client"]["temperature"],
            max_tokens=cfg["agents"]["client"]["max_tokens"],
        ),
        "diagnostician": DiagnosticianAgent(
            llm,
            system_prompt=get_diagnostician_prompt(lang),
            temperature=cfg["agents"]["diagnostician"]["temperature"],
            max_tokens=cfg["agents"]["diagnostician"]["max_tokens"],
        ),
        "auditor": EvidenceAuditorAgent(
            llm,
            system_prompt=get_auditor_prompt(lang),
            temperature=cfg["agents"]["auditor"]["temperature"],
            max_tokens=cfg["agents"]["auditor"]["max_tokens"],
        ),
    }

    # Initialise RAG pipeline (silent no-op if index has not been built)
    init_rag_pipeline()

    model_file = cfg["llm"]["model_file"]
    status = model_file if llm_available else "mock"
    return agents, status, llm_available


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_app_state() -> None:
    """Initialises Streamlit session_state keys on first page load."""
    if "agents_loaded" not in st.session_state:
        agents, status, llm_available = load_agents()
        AGENTS.update(agents)
        st.session_state.agents_loaded = True
        st.session_state.llm_status = status
        st.session_state.llm_available = llm_available

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver

        memory = MemorySaver()
        st.session_state.graph = build_graph(checkpointer=memory)
        st.session_state.config = {"configurable": {"thread_id": st.session_state.session_id}}

    st.session_state.setdefault("patient_mode", "Auto (Simulated Profile)")
    st.session_state.setdefault("session_language", "English")
    st.session_state.setdefault("started", False)


def _build_initial_state(language: str, interactive: bool, cfg: dict) -> dict:
    """Builds a complete SessionState dict for a fresh session."""
    return {
        "session_id": str(uuid.uuid4()),
        "client_profile": _default_profile(language),
        "transcript": [],
        "messages": [],
        "domains_covered": [],
        "domains_pending": [],  # init_session will shuffle and populate
        "coverage_complete": False,
        "retrieved_chunks": [],
        "query_history": [],
        "hypotheses": [],
        "audit_report": None,
        "risk_detected": False,
        "risk_type": None,
        "current_step": "",
        "turn_count": 0,
        "max_turns": cfg["session"]["max_turns"],
        "rapport_complete": False,
        "rapport_turns": 0,
        "rapport_turns_target": cfg["session"].get("rapport_turns", 3),
        "finalized": False,
        "interactive_mode": interactive,
        "language": language,
    }


def _default_profile(language: str) -> dict:
    """Returns a minimal synthetic profile used when no JSON profile is loaded."""
    if language == "Español":
        return {
            "profile_id": "demo_es",
            "demographics": {"name": "Demo", "age": 32, "gender": "no especificado"},
            "presenting_complaints": ["ansiedad", "dificultades para dormir", "irritabilidad"],
            "history": "Estrés laboral durante los últimos 4 meses. Sin historial psiquiátrico previo.",
            "language": "Español",
        }
    return {
        "profile_id": "demo_en",
        "demographics": {"name": "Demo", "age": 32, "gender": "unspecified"},
        "presenting_complaints": ["anxiety", "sleep difficulties", "irritability"],
        "history": "Work-related stress over the past 4 months. No prior psychiatric history.",
        "language": "English",
    }


# ---------------------------------------------------------------------------
# Graph execution helpers
# ---------------------------------------------------------------------------


def _char_stream(text: str) -> Generator[str, None, None]:
    """Yields individual characters with a small delay for a typing animation.

    Used with ``st.write_stream()`` to reveal therapist text progressively in
    interactive mode instead of showing the full response at once.
    """
    for char in text:
        yield char
        time.sleep(0.012)


def _run_until_interrupt(initial_state: dict | None = None, interactive: bool = False) -> None:
    """Streams the graph forward until it finishes or hits a human_input interrupt.

    Uses ``stream_mode='updates'`` so that each node's output is available as
    soon as the node completes.  Therapist and client messages are rendered
    immediately rather than after the full graph finishes.

    Args:
        initial_state: Full state dict when starting a new session; ``None``
            when resuming after a human-input interrupt.
        interactive: When ``True`` the therapist's message is revealed via a
            typing animation (``st.write_stream``); otherwise it is rendered
            as plain markdown.
    """
    graph = st.session_state.graph
    config = st.session_state.config
    status_placeholder = st.empty()

    try:
        for update in graph.stream(initial_state, config, stream_mode="updates"):
            node_name = next(iter(update))
            node_output = update[node_name]

            if node_name in ("therapist_ask", "rapport_ask"):
                status_placeholder.empty()
                transcript = node_output.get("transcript", [])
                if transcript:
                    last_msg = transcript[-1]
                    if last_msg.get("role") == "therapist":
                        with st.chat_message("assistant", avatar="🧑‍⚕️"):
                            if interactive:
                                st.write_stream(_char_stream(last_msg["content"]))
                            else:
                                st.markdown(last_msg["content"])
                            if domain := last_msg.get("domain"):
                                st.caption(f"{t('domain_caption')}: `{domain}`")
                status_placeholder = st.empty()

            elif node_name == "client_respond":
                status_placeholder.empty()
                transcript = node_output.get("transcript", [])
                if transcript:
                    last_msg = transcript[-1]
                    if last_msg.get("role") == "client":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(last_msg["content"])
                status_placeholder = st.empty()

            else:
                status_placeholder.markdown("⏳")

    except RuntimeError as exc:
        # LangGraph's IOExecutor submits checkpoint writes to a ThreadPoolExecutor.
        # In Streamlit's threaded script-runner model the executor can be torn down
        # before all pending put_writes() calls are submitted, raising this error.
        # By the time it surfaces, all node work and UI rendering are already done;
        # the failure is limited to the final bookkeeping checkpoint flush.
        if "cannot schedule new futures after shutdown" in str(exc):
            _logger.warning(
                "LangGraph checkpoint flush interrupted by executor shutdown "
                "(benign in Streamlit context — session data is intact): %s",
                exc,
            )
        else:
            raise
    finally:
        status_placeholder.empty()


def _snapshot() -> tuple[dict, bool]:
    """Returns (current_state_values, is_waiting_for_human_input)."""
    try:
        snap = st.session_state.graph.get_state(st.session_state.config)
        values = snap.values if snap.values else {}
        waiting = bool(snap.next) and snap.next[0] == "human_input"
    except Exception as exc:
        _logger.warning("get_state failed (returning empty snapshot): %s", exc)
        values = {}
        waiting = False
    return values, waiting


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------


def _render_sidebar(cfg: dict) -> None:
    with st.sidebar:
        st.header(t("sidebar_header"))

        # LLM / model status
        if st.session_state.get("llm_available"):
            st.success(f"{t('llm_loaded')}: `{st.session_state.llm_status}`")
        else:
            st.warning(t("llm_mock"))

        st.divider()

        # Language selector
        lang_options = ["English", "Español"]
        current_lang = st.session_state.session_language
        language = st.radio(
            t("language_label"),
            lang_options,
            index=lang_options.index(current_lang),
            disabled=st.session_state.started,
        )
        if language != current_lang:
            st.session_state.session_language = language
            st.rerun()

        # Mode selector — labels translate with the current language
        mode_auto = t("mode_auto")
        mode_interactive = t("mode_interactive")
        mode_options = [mode_auto, mode_interactive]
        current_mode = st.session_state.patient_mode
        # Normalise stored value to current-language label
        if current_mode not in mode_options:
            current_mode = mode_auto
        mode = st.radio(
            t("mode_label"),
            mode_options,
            index=mode_options.index(current_mode),
            disabled=st.session_state.started,
        )
        if mode != st.session_state.patient_mode:
            st.session_state.patient_mode = mode

        st.divider()
        max_turns = cfg["session"]["max_turns"]
        n_domains = len(cfg["session"]["domains"])
        st.caption(f"{max_turns} {t('max_turns_caption')} · {n_domains} {t('domains_caption')}")
        st.divider()

        if st.button(t("new_session_btn"), use_container_width=True):
            st.session_state.clear()
            st.rerun()


def _render_transcript(transcript: list[dict]) -> None:
    for msg in transcript:
        if msg["role"] == "therapist":
            with st.chat_message("assistant", avatar="🧑‍⚕️"):
                st.markdown(msg["content"])
                if domain := msg.get("domain"):
                    st.caption(f"{t('domain_caption')}: `{domain}`")
        elif msg["role"] == "client":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])


def _confidence_icon(confidence: str) -> str:
    return {"HIGH": "🟢", "ALTA": "🟢", "MEDIUM": "🟡", "MEDIA": "🟡"}.get(confidence, "🔴")


def _render_results(state: dict) -> None:
    if state.get("risk_detected"):
        from core.safety.risk_gate import RiskGate

        st.error(t("risk_terminated"))
        st.markdown(RiskGate().get_safe_response(state.get("risk_type", "")))
        return

    st.success(t("session_complete"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t("hypotheses_header"))
        hypotheses = state.get("hypotheses", [])
        if hypotheses:
            for h in hypotheses:
                confidence = h.get("confidence", "")
                icon = _confidence_icon(confidence)
                label = h.get("label", "?")
                code = h.get("code", "?")
                with st.expander(f"{icon} {label}  —  `{code}`"):
                    st.markdown(f"**{t('confidence_label')}:** {confidence}")
                    if h.get("evidence_for"):
                        st.markdown(f"**{t('evidence_for_label')}:**")
                        for e in h["evidence_for"]:
                            st.markdown(f"- {e}")
                    if h.get("evidence_against"):
                        st.markdown(f"**{t('evidence_against_label')}:**")
                        for e in h["evidence_against"]:
                            st.markdown(f"- {e}")
        else:
            st.info(t("no_hypotheses"))

    with col2:
        st.subheader(t("audit_header"))
        audit = state.get("audit_report") or {}
        score = audit.get("traceability_score", 1.0)
        st.metric(t("traceability_metric"), f"{score:.0%}")
        issues = audit.get("issues", [])
        if issues:
            st.warning(t("issues_warning", n=len(issues)))
            for issue in issues:
                st.caption(f"• [{issue.get('hypothesis', '?')}] {issue.get('claim', '')}")
        else:
            st.success(t("no_issues"))
        if commentary := audit.get("llm_commentary"):
            st.markdown(f"**{t('auditor_comment')}:**")
            st.markdown(f"> {commentary}")

    st.subheader(t("full_transcript"))
    with st.expander(t("view_transcript")):
        st.json(state.get("transcript", []))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _load_config()

    st.set_page_config(
        page_title=t("page_title"),
        page_icon="🧠",
        layout="wide",
    )

    _init_app_state()
    _render_sidebar(cfg)

    st.title(f"🧠 {t('page_title')}")
    st.caption(t("page_subtitle"))
    st.warning(t("disclaimer"))
    st.divider()

    current_state, waiting_for_human = _snapshot()
    transcript = current_state.get("transcript", [])
    _render_transcript(transcript)

    # --- Session finalised ---
    if current_state.get("finalized"):
        _render_results(current_state)
        return

    # --- Session not yet started ---
    if not st.session_state.started:
        lang = st.session_state.session_language
        mode = st.session_state.patient_mode
        is_interactive = t("mode_interactive") in mode

        btn_label = t("btn_start_interactive") if is_interactive else t("btn_start_auto")
        if st.button(btn_label, type="primary"):
            st.session_state.started = True
            initial_state = _build_initial_state(lang, is_interactive, cfg)
            _run_until_interrupt(initial_state, interactive=is_interactive)
            st.rerun()
        return

    # --- Interactive mode: waiting for human input ---
    if waiting_for_human:
        user_input = st.chat_input(t("chat_placeholder"))
        if user_input:
            updated_transcript = list(current_state.get("transcript", []))
            updated_transcript.append(
                {
                    "role": "client",
                    "content": user_input,
                    "turn_id": len(updated_transcript),
                }
            )
            st.session_state.graph.update_state(
                st.session_state.config,
                {"transcript": updated_transcript, "current_step": "human_input"},
                as_node="human_input",
            )
            _run_until_interrupt(interactive=True)
            st.rerun()
        return

    # --- Auto mode: keep stepping until done ---
    if st.session_state.started and not current_state.get("finalized"):
        turn = current_state.get("turn_count", 0)
        max_turns = current_state.get("max_turns", cfg["session"]["max_turns"])
        pending = len(current_state.get("domains_pending", []))
        covered = len(current_state.get("domains_covered", []))
        st.info(t("progress_info", turn=turn, max=max_turns, pending=pending, covered=covered))

        _run_until_interrupt()
        st.rerun()


if __name__ == "__main__":
    main()
