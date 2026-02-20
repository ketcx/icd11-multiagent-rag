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
        language = state.get("language", "Español")
        profile = state.get("client_profile", {})
        messages = self._build_messages(state["transcript"], profile, language)
        response = self._generate(messages)

        state["transcript"].append(
            {
                "role": "client",
                "content": response,
                "turn_id": len(state["transcript"]),
            }
        )

        return state

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
                "Responde a la última pregunta del terapeuta manteniéndote en tu personaje. "
                "Responde en español de forma natural y breve (1-3 oraciones). "
                "No uses jerga clínica para describir tus síntomas."
            )
        else:
            instruction = (
                "Respond to the therapist's last question staying in character. "
                "Reply in English naturally and briefly (1-3 sentences). "
                "Avoid clinical jargon when describing your symptoms."
            )
        messages.append({"role": "user", "content": instruction})

        return messages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_profile(profile: dict, language: str) -> str:
    """Converts a profile dict into a human-readable context string for the LLM."""
    demographics = profile.get("demographics", {})
    name = demographics.get("name", "Unknown")
    age = demographics.get("age", "?")
    gender = demographics.get("gender", "")
    complaints = profile.get("presenting_complaints", [])
    history = profile.get("history", "")

    if language == "Español":
        complaints_str = ", ".join(complaints) if complaints else "no especificadas"
        parts = [
            f"Eres {name}, {age} años{', ' + gender if gender else ''}.",
            f"Motivo de consulta: {complaints_str}.",
        ]
        if history:
            parts.append(f"Historial: {history}")
        parts.append(
            "Durante la entrevista, responde de forma natural y espontánea. "
            "No menciones tu perfil directamente; deja que la información emerja gradualmente."
        )
    else:
        complaints_str = ", ".join(complaints) if complaints else "unspecified"
        parts = [
            f"You are {name}, {age} years old{', ' + gender if gender else ''}.",
            f"Presenting complaints: {complaints_str}.",
        ]
        if history:
            parts.append(f"Background: {history}")
        parts.append(
            "During the interview, respond naturally and spontaneously. "
            "Do not mention your profile directly; let the information emerge gradually."
        )

    return " ".join(parts)
