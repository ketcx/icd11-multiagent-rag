"""Client agent responsible for simulating a synthetic patient."""

from __future__ import annotations

from core.agents.base import BaseAgent


class ClientAgent(BaseAgent):
    """Simulates a patient response grounded in an assigned profile.

    The client profile is injected at the start of every message chain so
    the LLM understands who it is portraying — demographics, presenting
    complaints, and clinical history all influence the generated response,
    producing unique conversations across sessions even with the same profile.
    """

    def act(self, state: dict) -> dict:
        """Generates the client's response to the last therapist question.

        Args:
            state: Current ``SessionState`` with transcript and client_profile.

        Returns:
            Partial state dict with the updated transcript.
        """
        from core.agents.prompts import get_client_prompt

        language = state.get("language", "Español")
        profile = state.get("client_profile", {})
        messages = self._build_messages(state["transcript"], profile, language)
        response = self._generate(messages, system_prompt=get_client_prompt(language))

        state["transcript"].append(
            {
                "role": "client",
                "content": response,
                "turn_id": len(state["transcript"]),
            }
        )

        return state

    def act_stream(self, state: dict):
        """Yields tokens for the client's response without updating state.

        Mirrors the prompt construction of ``act()`` so the streamed output is
        identical to what would be generated non-streamingly.

        Args:
            state: Current ``SessionState`` dictionary.

        Yields:
            Non-empty token strings from the model, or nothing when LLM is
            unavailable (mock mode).
        """
        from core.agents.prompts import get_client_prompt

        language = state.get("language", "Español")
        profile = state.get("client_profile", {})
        messages = self._build_messages(state["transcript"], profile, language)
        yield from self._generate_stream(messages, system_prompt=get_client_prompt(language))

    def _build_messages(self, transcript: list[dict], profile: dict, language: str) -> list[dict]:
        """Builds the message payload for the LLM.

        Injects the client profile as an opening exchange so the model can
        ground its response in the patient's demographics, complaints, and
        history.  The transcript is then replayed in alternating user/assistant
        roles before the final instruction.

        Args:
            transcript: Full conversation so far.
            profile: Client profile dict (demographics, complaints, history).
            language: Language for the response ("Español" or "English").

        Returns:
            List of chat-completion message dicts.
        """
        messages: list[dict] = []

        # --- Profile context injection ---
        if profile:
            profile_context = _format_profile(profile, language)
            messages.append({"role": "user", "content": profile_context})
            if language == "Español":
                ack = "Entendido. Voy a encarnar a este personaje durante la entrevista."
            else:
                ack = "Understood. I will embody this character throughout the interview."
            messages.append({"role": "assistant", "content": ack})

        # --- Conversation history ---
        for turn in transcript:
            role = "assistant" if turn["role"] == "client" else "user"
            messages.append({"role": role, "content": turn["content"]})

        # --- Final instruction ---
        if language == "Español":
            instruction = (
                "Responde AHORA a la última pregunta del terapeuta como el paciente. "
                "Escribe SOLO las palabras que el paciente diría: 1-3 oraciones naturales en español. "
                "USA SIEMPRE la primera persona (yo). NUNCA uses tercera persona para referirte a ti mismo. "
                "PROHIBIDO: etiquetas de rol (user:, model:, roleplaying, etc.), "
                "acotaciones, meta-texto, etiquetas XML/HTML (<|...), "
                "o cualquier cosa que no sea el diálogo del paciente."
            )
        else:
            instruction = (
                "Respond NOW to the therapist's last question as the patient. "
                "Write ONLY the patient's spoken words: 1-3 natural sentences in English. "
                "ALWAYS use first person (I/me/my). NEVER refer to yourself in third person. "
                "FORBIDDEN: role labels (user:, model:, roleplaying, etc.), "
                "stage directions, meta-text, XML/HTML tags (<|...), "
                "or anything other than the patient's dialogue."
            )
        messages.append({"role": "user", "content": instruction})

        return messages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Spanish names that end in 'a' but are masculine — expand as needed.
_MASC_A_EXCEPTIONS = frozenset({"borja", "luca", "iker", "joseba", "nikita"})

# Common feminine name endings in Spanish/English that reliably signal gender.
_FEM_ENDINGS = ("a", "ia", "na", "ra", "la", "sa", "da", "ea", "ía", "ina", "elle", "ette")
_MASC_ENDINGS = ("o", "os", "ón", "él", "el", "al", "or", "án", "us", "is")


def _infer_gender(name: str, declared: str, language: str) -> str:
    """Resolves the effective gender for the client profile.

    If *declared* already carries a clear gender label it is returned as-is.
    Otherwise the name is used as a heuristic fallback.  The result is a
    human-readable label in the correct language for insertion into the prompt.
    """
    declared_lower = declared.lower().strip()

    # Already has a real gender value
    if declared_lower in ("masculino", "male", "hombre", "man", "m"):
        return "masculino" if language == "Español" else "male"
    if declared_lower in ("femenino", "female", "mujer", "woman", "f"):
        return "femenino" if language == "Español" else "female"

    # Heuristic: derive from first name
    first = name.split()[0].lower().strip()
    if first in _MASC_A_EXCEPTIONS:
        inferred = "masculino"
    elif any(first.endswith(e) for e in _FEM_ENDINGS):
        inferred = "femenino"
    elif any(first.endswith(e) for e in _MASC_ENDINGS):
        inferred = "masculino"
    else:
        inferred = None  # Cannot infer — omit gender instruction

    if inferred is None:
        return ""
    return inferred if language == "Español" else ("female" if inferred == "femenino" else "male")


def _gender_instruction(gender_label: str, name: str, language: str) -> str:
    """Returns an explicit grammatical-gender instruction for the LLM prompt."""
    if not gender_label:
        return ""
    if language == "Español":
        if gender_label == "femenino":
            return (
                f"Tu nombre es {name} y eres mujer. "
                "Usa concordancia femenina en todos tus adjetivos y participios "
                "(p. ej. 'estoy cansada', 'me siento agotada', 'estoy preocupada')."
            )
        return (
            f"Tu nombre es {name} y eres hombre. "
            "Usa concordancia masculina en todos tus adjetivos y participios "
            "(p. ej. 'estoy cansado', 'me siento agotado', 'estoy preocupado')."
        )
    else:
        if gender_label == "female":
            return f"Your name is {name} and you are a woman. Use she/her self-references if any."
        return f"Your name is {name} and you are a man. Use he/him self-references if any."


def _format_profile(profile: dict, language: str) -> str:
    """Converts a profile dict into a human-readable context string for the LLM."""
    demographics = profile.get("demographics", {})
    name = demographics.get("name", "Unknown")
    age = demographics.get("age", "?")
    declared_gender = demographics.get("gender", "")
    complaints = profile.get("presenting_complaints", [])
    history = profile.get("history", "")

    gender_label = _infer_gender(name, declared_gender, language)
    gender_instr = _gender_instruction(gender_label, name, language)

    if language == "Español":
        complaints_str = ", ".join(complaints) if complaints else "no especificadas"
        parts = [
            f"Eres {name}, {age} años.",
            f"Motivo de consulta: {complaints_str}.",
        ]
        if history:
            parts.append(f"Historial: {history}")
        if gender_instr:
            parts.append(gender_instr)
        parts.append(
            "Durante la entrevista, responde de forma natural y espontánea en primera persona. "
            "No menciones tu perfil directamente; deja que la información emerja gradualmente."
        )
    else:
        complaints_str = ", ".join(complaints) if complaints else "unspecified"
        parts = [
            f"You are {name}, {age} years old.",
            f"Presenting complaints: {complaints_str}.",
        ]
        if history:
            parts.append(f"Background: {history}")
        if gender_instr:
            parts.append(gender_instr)
        parts.append(
            "During the interview, respond naturally and spontaneously in first person. "
            "Do not mention your profile directly; let the information emerge gradually."
        )

    return " ".join(parts)
