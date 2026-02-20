"""Node functions for the LangGraph graph."""

from __future__ import annotations

import random

from core.orchestration.state import SessionState

AGENTS: dict = {}  # Global registry for initialized agents

# ---------------------------------------------------------------------------
# Mock response banks — domain-aware, varied per call via random.choice
# ---------------------------------------------------------------------------

_MOCK_THERAPIST_ES: dict[str, list[str]] = {
    "mood": [
        "[Mock] ¿Cómo describirías tu estado de ánimo en las últimas semanas?",
        "[Mock] ¿Has notado cambios en cómo te sientes emocionalmente?",
        "[Mock] ¿Hay momentos del día en que te sientes más decaído o sin energía?",
    ],
    "anxiety": [
        "[Mock] ¿Experimentas preocupaciones que te resultan difíciles de controlar?",
        "[Mock] ¿Con qué frecuencia sientes tensión o nerviosismo sin una causa clara?",
        "[Mock] ¿Hay situaciones concretas que te generen mucha angustia?",
    ],
    "sleep": [
        "[Mock] ¿Cómo estás durmiendo últimamente? ¿Tienes dificultades para conciliar el sueño?",
        "[Mock] ¿Te despiertas durante la noche o muy temprano sin poder volver a dormir?",
        "[Mock] ¿Sientes que el sueño te resulta reparador?",
    ],
    "eating": [
        "[Mock] ¿Has notado cambios en tu apetito o en tus hábitos alimenticios?",
        "[Mock] ¿Has perdido o ganado peso en los últimos meses?",
        "[Mock] ¿Tu relación con la comida ha cambiado de alguna manera?",
    ],
    "substances": [
        "[Mock] ¿Consumes alcohol u otras sustancias? ¿Con qué frecuencia?",
        "[Mock] ¿Has recurrido a alguna sustancia para manejar el estrés o el malestar?",
    ],
    "psychosis": [
        "[Mock] ¿Has tenido experiencias inusuales, como escuchar o ver cosas que otros no perciben?",
        "[Mock] ¿Has tenido pensamientos que te parezcan extraños o difíciles de explicar?",
    ],
    "trauma": [
        "[Mock] ¿Has vivido alguna experiencia que te haya resultado especialmente difícil o traumática?",
        "[Mock] ¿Hay recuerdos que aparecen de forma intrusiva y te generan malestar?",
    ],
    "ocd": [
        "[Mock] ¿Tienes pensamientos repetitivos que te resultan difíciles de controlar?",
        "[Mock] ¿Realizas algún comportamiento de forma repetida para aliviar la ansiedad?",
    ],
    "cognition": [
        "[Mock] ¿Has notado cambios en tu memoria, concentración o capacidad para tomar decisiones?",
        "[Mock] ¿Te cuesta más de lo habitual enfocarte en una tarea?",
    ],
    "social_functioning": [
        "[Mock] ¿Cómo están tus relaciones con familia, amigos o compañeros?",
        "[Mock] ¿Has reducido tus actividades sociales o te has aislado últimamente?",
    ],
    "suicidal_ideation": [
        "[Mock] ¿Has tenido pensamientos de hacerte daño o de que estarías mejor muerto?",
        "[Mock] En los momentos más difíciles, ¿has pensado en quitarte la vida?",
    ],
}

_MOCK_THERAPIST_EN: dict[str, list[str]] = {
    "mood": [
        "[Mock] How would you describe your mood over the past few weeks?",
        "[Mock] Have you noticed changes in how you feel emotionally day to day?",
        "[Mock] Are there times when you feel particularly low or lack energy?",
    ],
    "anxiety": [
        "[Mock] Do you experience worries that are difficult to control?",
        "[Mock] How often do you feel tense or nervous without a clear reason?",
        "[Mock] Are there specific situations that cause you a lot of distress?",
    ],
    "sleep": [
        "[Mock] How has your sleep been lately? Do you have trouble falling asleep?",
        "[Mock] Do you wake up during the night or very early and can't go back to sleep?",
        "[Mock] Do you feel rested after a night's sleep?",
    ],
    "eating": [
        "[Mock] Have you noticed any changes in your appetite or eating habits?",
        "[Mock] Have you lost or gained weight recently?",
        "[Mock] Has your relationship with food changed in any way?",
    ],
    "substances": [
        "[Mock] Do you use alcohol or other substances? How often?",
        "[Mock] Have you used any substance to cope with stress or discomfort?",
    ],
    "psychosis": [
        "[Mock] Have you had any unusual experiences, like hearing or seeing things others don't?",
        "[Mock] Have you had thoughts that feel strange or hard to explain?",
    ],
    "trauma": [
        "[Mock] Have you been through any particularly difficult or traumatic experiences?",
        "[Mock] Are there memories that intrude on your thoughts and cause distress?",
    ],
    "ocd": [
        "[Mock] Do you have repetitive thoughts that are hard to control?",
        "[Mock] Do you perform any behaviours repeatedly to relieve anxiety?",
    ],
    "cognition": [
        "[Mock] Have you noticed changes in your memory, concentration, or ability to make decisions?",
        "[Mock] Is it harder than usual to focus on a task?",
    ],
    "social_functioning": [
        "[Mock] How are your relationships with family, friends, or colleagues?",
        "[Mock] Have you been withdrawing from social activities lately?",
    ],
    "suicidal_ideation": [
        "[Mock] Have you had thoughts of hurting yourself or that you'd be better off dead?",
        "[Mock] In your hardest moments, have you thought about ending your life?",
    ],
}

_MOCK_CLIENT_ES: dict[str, list[str]] = {
    "mood": [
        "La verdad es que me he sentido muy apagado, sin ganas de hacer nada.",
        "Hay días en que me levanto y todo parece gris. Antes disfrutaba más las cosas.",
        "Me siento triste sin razón concreta, y eso me preocupa bastante.",
    ],
    "anxiety": [
        "Sí, me preocupo muchísimo por el trabajo y por mi familia, aunque no haya motivo real.",
        "Siento como un nudo en el estómago casi todo el tiempo. Es agotador.",
        "Me pongo muy nervioso en situaciones cotidianas que antes no me afectaban.",
    ],
    "sleep": [
        "Duermo fatal. Me cuesta horas quedarme dormido y luego me despierto a las cuatro de la mañana.",
        "No descanso bien. Me desvelo pensando en mil cosas y por la mañana estoy agotado.",
        "A veces duermo demasiado, pero aun así me siento sin energía durante el día.",
    ],
    "eating": [
        "He perdido el apetito casi por completo. A veces se me olvida comer.",
        "Como más de lo habitual cuando estoy estresado, como para calmarme.",
        "Mi apetito está bien, aunque noto que he perdido un poco de peso sin proponérmelo.",
    ],
    "substances": [
        "Bebo un par de cervezas por las noches para poder relajarme.",
        "Fumo más cuando estoy estresado, y a veces tomo algún ansiolítico que me sobró.",
        "No consumo nada especial, solo cafeína en exceso.",
    ],
    "psychosis": [
        "No, nada de eso. Solo que mi mente no para.",
        "A veces me parece escuchar que me llaman y cuando miro no hay nadie, pero creo que es el cansancio.",
        "Tengo pensamientos muy acelerados pero nada inusual como alucinaciones.",
    ],
    "trauma": [
        "Sí, pasé por una situación muy difícil hace unos años que no he superado del todo.",
        "Tengo recuerdos de una época muy mala que aparecen de repente y me ponen mal.",
        "No creo que haya tenido nada traumático, simplemente mucho estrés acumulado.",
    ],
    "ocd": [
        "Reviso mucho las cosas: si cerré el gas, si apagué la luz. Sé que es exagerado.",
        "Tengo pensamientos que se repiten solos y que me cuestan mucho quitarme de la cabeza.",
        "No tengo rituales especiales, pero sí soy muy perfeccionista y eso me genera mucha tensión.",
    ],
    "cognition": [
        "Me cuesta mucho concentrarme. Leo un párrafo y tengo que volver a empezar.",
        "Se me olvidan cosas que antes recordaba sin problema. Me preocupa.",
        "Mi mente va muy lenta, como si tuviera niebla. Tomo decisiones muy difícil.",
    ],
    "social_functioning": [
        "Me he apartado bastante. Ya no quedo con amigos como antes.",
        "En el trabajo me cuesta relacionarme. Prefiero encerrarme en mis tareas.",
        "Mi pareja dice que estoy más distante, y tiene razón.",
    ],
    "suicidal_ideation": [
        "A veces pienso que sería mejor no estar aquí, aunque no tengo ningún plan.",
        "Hay momentos muy oscuros, pero no llegaría a hacerme daño.",
        "No, nunca he pensado en eso. Solo quiero que este malestar se acabe.",
    ],
}

_MOCK_CLIENT_EN: dict[str, list[str]] = {
    "mood": [
        "Honestly, I've been feeling very down. I can't find motivation for anything.",
        "Most days feel grey. I used to enjoy things more than I do now.",
        "I feel sad without a clear reason, which worries me.",
    ],
    "anxiety": [
        "Yes, I worry a lot about work and family even when there's no real reason.",
        "I have a constant knot in my stomach. It's exhausting.",
        "I get very anxious in everyday situations that didn't used to bother me.",
    ],
    "sleep": [
        "I sleep terribly. It takes hours to fall asleep and then I wake up at four in the morning.",
        "I don't rest well. I lie awake thinking and I'm exhausted in the morning.",
        "Sometimes I sleep too much but still feel drained during the day.",
    ],
    "eating": [
        "I've almost completely lost my appetite. I sometimes forget to eat.",
        "I eat more than usual when I'm stressed, as a way to calm down.",
        "My appetite is okay, though I've lost a bit of weight without trying.",
    ],
    "substances": [
        "I have a couple of beers in the evenings to wind down.",
        "I smoke more when I'm stressed, and sometimes I take a leftover anxiolytic.",
        "Nothing special, just too much caffeine.",
    ],
    "psychosis": [
        "No, nothing like that. My mind just doesn't stop.",
        "Sometimes I think I hear someone calling my name and when I look there's no one — probably just tiredness.",
        "I have racing thoughts but nothing unusual like hallucinations.",
    ],
    "trauma": [
        "Yes, I went through something really difficult a few years ago that I haven't fully moved past.",
        "I have memories of a very hard period that suddenly come back and affect me.",
        "I don't think I've had anything traumatic, just a lot of accumulated stress.",
    ],
    "ocd": [
        "I check things a lot — whether I turned off the gas, locked the door. I know it's excessive.",
        "I have thoughts that repeat on their own and are really hard to shake.",
        "No special rituals, but I'm very perfectionistic and that creates a lot of tension.",
    ],
    "cognition": [
        "It's hard to concentrate. I read a paragraph and have to start over.",
        "I forget things I used to remember easily. It worries me.",
        "My mind feels foggy and slow. Making decisions is very hard.",
    ],
    "social_functioning": [
        "I've withdrawn a lot. I don't see friends the way I used to.",
        "At work I struggle to engage with others. I prefer to just focus on my tasks.",
        "My partner says I've been more distant, and they're right.",
    ],
    "suicidal_ideation": [
        "Sometimes I think it'd be better not to be here, though I have no plan.",
        "There are very dark moments, but I wouldn't actually hurt myself.",
        "No, I've never thought about that. I just want this pain to stop.",
    ],
}


def _mock_therapist_question(domain: str, language: str) -> str:
    """Returns a random domain-appropriate mock therapist question."""
    bank = _MOCK_THERAPIST_ES if language == "Español" else _MOCK_THERAPIST_EN
    options = bank.get(domain, [f"[Mock] ¿Puedes hablarme sobre {domain}?"])
    return random.choice(options)


def _mock_client_response(domain: str, language: str) -> str:
    """Returns a random domain-appropriate mock client response."""
    bank = _MOCK_CLIENT_ES if language == "Español" else _MOCK_CLIENT_EN
    options = bank.get(domain, ["[Mock] No sé bien cómo explicarlo."])
    return random.choice(options)


def _current_domain(state: SessionState) -> str:
    """Infers the domain being addressed from the last therapist turn."""
    for turn in reversed(state.get("transcript", [])):
        if turn.get("role") == "therapist" and turn.get("domain"):
            return turn["domain"]
    pending = state.get("domains_pending", [])
    return pending[0] if pending else "mood"


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def init_session(state: SessionState) -> dict:
    """Initialises the session: shuffles domain order and resets control flags.

    Domain order is randomised on every new session so that even the same
    client profile yields a different interview flow and, consequently, a
    different diagnostic path.
    """
    from core.agents.therapist import TherapistAgent

    # Only shuffle when starting fresh (domains_pending is empty)
    if not state.get("domains_pending"):
        domains = TherapistAgent.DOMAINS.copy()
        random.shuffle(domains)
    else:
        domains = state["domains_pending"]

    return {
        "domains_pending": domains,
        "domains_covered": state.get("domains_covered", []),
        "transcript": state.get("transcript", []),
        "turn_count": state.get("turn_count", 0),
        "coverage_complete": False,
        "risk_detected": False,
        "finalized": False,
        "current_step": "init_session",
    }


def therapist_ask(state: SessionState) -> dict:
    """Generates the therapist's next question for the current pending domain."""
    if "therapist" in AGENTS:
        updated_state = AGENTS["therapist"].act(state)
        updated_state["current_step"] = "therapist_ask"
        updated_state["turn_count"] = state.get("turn_count", 0) + 1
        return updated_state

    # Mock fallback — domain-aware, picks a random question from the bank
    transcript = state.get("transcript", [])
    turn_count = state.get("turn_count", 0)
    language = state.get("language", "Español")
    pending = state.get("domains_pending", [])
    domain = pending[0] if pending else "mood"

    content = _mock_therapist_question(domain, language)

    transcript.append(
        {
            "role": "therapist",
            "content": content,
            "domain": domain,
            "turn_id": len(transcript),
        }
    )
    return {
        "transcript": transcript,
        "turn_count": turn_count + 1,
        "current_step": "therapist_ask",
    }


def client_respond(state: SessionState) -> dict:
    """Generates the client's response based on their profile."""
    if "client" in AGENTS:
        updated_state = AGENTS["client"].act(state)
        updated_state["current_step"] = "client_respond"
        return updated_state

    # Mock fallback — infers the domain from the last therapist turn
    transcript = state.get("transcript", [])
    language = state.get("language", "Español")
    domain = _current_domain(state)

    content = _mock_client_response(domain, language)

    transcript.append(
        {
            "role": "client",
            "content": content,
            "turn_id": len(transcript),
        }
    )
    return {
        "transcript": transcript,
        "current_step": "client_respond",
    }


def human_input_node(state: SessionState) -> dict:
    """Waits for human input during an interactive session.

    The Streamlit UI injects the human message into the state transcript before
    resuming the graph; this node simply marks the step as executed.
    """
    return {"current_step": "human_input"}


def coverage_check(state: SessionState) -> dict:
    """Determines whether all clinical domains have been covered.

    Marks coverage complete when every domain in the session config has been
    addressed or when the turn ceiling is reached, preventing infinite loops.
    """
    from core.agents.therapist import TherapistAgent

    transcript = state.get("transcript", [])
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 40)

    # Derive covered domains from transcript entries that carry a domain tag
    covered = list(state.get("domains_covered", []))
    for entry in transcript:
        domain = entry.get("domain")
        if domain and domain not in covered:
            covered.append(domain)

    all_domains = TherapistAgent.DOMAINS
    pending = [d for d in state.get("domains_pending", all_domains) if d not in covered]
    coverage_complete = (not pending) or (turn_count >= max_turns)

    return {
        "domains_covered": covered,
        "domains_pending": pending,
        "coverage_complete": coverage_complete,
    }


def risk_check(state: SessionState) -> dict:
    """Runs the RiskGate over the most recent transcript turn.

    Checks only the latest entry to avoid repeated classification of historical
    turns on every pass through the node.
    """
    from core.safety.risk_gate import RiskGate

    transcript = state.get("transcript", [])
    if not transcript:
        return {"risk_detected": False, "risk_type": None}

    latest_text = transcript[-1].get("content", "")
    gate = RiskGate()
    is_risky, risk_type = gate.check(latest_text)

    return {"risk_detected": is_risky, "risk_type": risk_type}


def retrieve_context(state: SessionState) -> dict:
    """Executes RAG: constructs queries → hybrid retrieval."""
    from core.retrieval import get_rag_pipeline

    rag_pipeline = get_rag_pipeline()
    transcript = state.get("transcript", [])
    language = state.get("language", "Español")

    retrieved_chunks: list[dict] = []
    query_history: list[dict] = []

    if rag_pipeline and len(transcript) > 0:
        try:
            queries = rag_pipeline["query_builder"].build_queries(transcript)

            if queries.get("semantic"):
                semantic_chunks = rag_pipeline["retriever"].retrieve(queries["semantic"])
                retrieved_chunks.extend(semantic_chunks)
                query_history.append(
                    {
                        "type": "semantic",
                        "query": queries["semantic"],
                        "results": len(semantic_chunks),
                    }
                )

            if queries.get("exact"):
                for exact_query in queries["exact"]:
                    exact_chunks = rag_pipeline["retriever"].retrieve(exact_query)
                    retrieved_chunks.extend(exact_chunks)
                query_history.append(
                    {
                        "type": "exact",
                        "queries": queries["exact"],
                        "total_results": len(retrieved_chunks),
                    }
                )

            # Dedup by content
            seen: set[str] = set()
            deduped: list[dict] = []
            for chunk in retrieved_chunks:
                content = chunk.get("content", "")
                if content not in seen:
                    seen.add(content)
                    deduped.append(chunk)
            retrieved_chunks = deduped[:6]

        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "RAG retrieval failed: %s — using mock context.", exc
            )

    # Fallback mock chunks when RAG is unavailable
    if not retrieved_chunks:
        if language == "Español":
            retrieved_chunks = [
                {
                    "content": "Trastorno Depresivo — CIE-11 6A70: Un episodio depresivo se caracteriza por tristeza persistente y pérdida de interés en actividades.",
                    "metadata": {"code": "6A70"},
                    "source": "mock",
                },
                {
                    "content": "Trastorno de Ansiedad — CIE-11 6B00: Ansiedad excesiva y preocupación persistente que interfieren con el funcionamiento diario.",
                    "metadata": {"code": "6B00"},
                    "source": "mock",
                },
            ]
        else:
            retrieved_chunks = [
                {
                    "content": "Depressive Episode — ICD-11 6A70: A depressive episode is characterised by persistent low mood and loss of interest in activities.",
                    "metadata": {"code": "6A70"},
                    "source": "mock",
                },
                {
                    "content": "Anxiety Disorder — ICD-11 6B00: Excessive anxiety and persistent worry that interfere with daily functioning.",
                    "metadata": {"code": "6B00"},
                    "source": "mock",
                },
            ]

    return {
        "retrieved_chunks": retrieved_chunks,
        "query_history": query_history,
    }


def diagnostician_draft(state: SessionState) -> dict:
    """Generates diagnostic hypotheses via the Diagnostician agent."""
    if "diagnostician" in AGENTS:
        updated_state = AGENTS["diagnostician"].act(state)
        updated_state["current_step"] = "diagnostician_draft"
        return updated_state

    # Mock fallback
    language = state.get("language", "Español")
    domains_covered = state.get("domains_covered", [])

    if language == "Español":
        label = "Trastorno de Ansiedad Generalizada (simulado)"
        code = "6B00"
        evidence = [d for d in domains_covered if d in ("anxiety", "sleep", "mood")]
        hypotheses = [
            {
                "label": label,
                "code": code,
                "confidence": "MEDIA" if len(evidence) < 2 else "ALTA",
                "evidence_for": evidence or ["síntomas reportados en la entrevista"],
                "evidence_against": [],
            }
        ]
    else:
        label = "Generalised Anxiety Disorder (simulated)"
        code = "6B00"
        evidence = [d for d in domains_covered if d in ("anxiety", "sleep", "mood")]
        hypotheses = [
            {
                "label": label,
                "code": code,
                "confidence": "MEDIUM" if len(evidence) < 2 else "HIGH",
                "evidence_for": evidence or ["symptoms reported in the interview"],
                "evidence_against": [],
            }
        ]

    return {
        "hypotheses": hypotheses,
        "current_step": "diagnostician_draft",
    }


def evidence_audit(state: SessionState) -> dict:
    """Audits hypotheses by verifying claim traceability."""
    if "auditor" in AGENTS:
        updated_state = AGENTS["auditor"].act(state)
        updated_state["current_step"] = "evidence_audit"
        return updated_state

    return {
        "audit_report": {
            "verified": True,
            "traceability_score": 1.0,
            "total_claims": 0,
            "grounded_claims": 0,
            "issues": [],
            "llm_commentary": None,
        },
        "current_step": "evidence_audit",
    }


def finalize_session(state: SessionState) -> dict:
    """Generates the final structured output."""
    return {"finalized": True, "current_step": "finalized"}


def safe_exit(state: SessionState) -> dict:
    """Safe exit path triggered by RiskGate."""
    return {
        "finalized": True,
        "current_step": "safe_exit",
        "risk_detected": True,
    }
