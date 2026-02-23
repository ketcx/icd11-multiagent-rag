"""Therapist agent responsible for interviewing the client."""

from __future__ import annotations

import random

from core.agents.base import BaseAgent
from core.agents.prompts import get_rapport_prompt

# Random name pools used when no real patient name is available in the profile.
_RANDOM_NAMES_ES = [
    "Carlos", "Lucía", "Andrés", "Sofía", "Miguel", "Elena",
    "Javier", "Marta", "Roberto", "Carmen", "Pablo", "Laura",
]
_RANDOM_NAMES_EN = [
    "Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey",
    "Riley", "Jamie", "Quinn", "Avery", "Dana", "Skyler",
]

_GENERIC_NAMES = {"demo", "unknown", "test", "test user", "paciente", "patient", ""}


def _resolve_patient_name(state: dict, language: str) -> tuple[str, bool]:
    """Returns (first_name, was_randomly_assigned).

    Extracts the first name from ``client_profile.demographics.name``.
    If the stored name is generic/absent, picks a random one from the pool so
    the therapist can address the patient consistently throughout the session.
    """
    profile = state.get("client_profile", {})
    raw_name: str = profile.get("demographics", {}).get("name", "").strip()
    # Use only the first token so "María García" becomes "María"
    first_name = raw_name.split()[0] if raw_name else ""

    if first_name.lower() in _GENERIC_NAMES:
        pool = _RANDOM_NAMES_ES if language == "Español" else _RANDOM_NAMES_EN
        return random.choice(pool), True

    return first_name, False


class TherapistAgent(BaseAgent):
    """Explores clinical domains empathetically without diagnosing."""

    # Target domains to evaluate
    DOMAINS = [
        "mood",
        "anxiety",
        "sleep",
        "eating",
        "substances",
        "psychosis",
        "trauma",
        "ocd",
        "cognition",
        "social_functioning",
        "suicidal_ideation",
    ]

    def act(self, state: dict) -> dict:
        """Generates the next therapist question.

        1. Review ``domains_pending`` from state (preserves session-shuffled order).
        2. Select next pending domain.
        3. Generate empathetic question.
        4. Update state with turn + tracking.

        Using ``state["domains_pending"]`` rather than recomputing from the
        class-level ``DOMAINS`` list ensures that the domain order shuffled in
        ``init_session`` is honoured throughout the session, producing a
        different interview flow every run.
        """
        # Honour the session-level order set by init_session (may be shuffled).
        # Fall back to filtering DOMAINS only when domains_pending is absent.
        pending = state.get("domains_pending") or [
            d for d in self.DOMAINS if d not in state.get("domains_covered", [])
        ]

        if not pending:
            state["coverage_complete"] = True
            return state

        next_domain = pending[0]
        language = state.get("language", "Español")

        # Build prompt adding history + target domain logic
        messages = self._build_messages(state["transcript"], next_domain, language)
        response = self._generate(messages)

        # Update transcript
        state["transcript"].append(
            {
                "role": "therapist",
                "content": response,
                "domain": next_domain,
                "turn_id": len(state["transcript"]),
            }
        )

        # Explicitly set the pending domains for external tracking
        state["domains_pending"] = pending

        return state

    def act_stream(self, state: dict):
        """Yields tokens for the therapist's next question without updating state.

        Mirrors the prompt construction of ``act()`` so the streamed output is
        identical to what would be generated non-streamingly.

        Args:
            state: Current ``SessionState`` dictionary.

        Yields:
            Non-empty token strings from the model, or nothing when LLM is
            unavailable (mock mode).
        """
        pending = state.get("domains_pending") or [
            d for d in self.DOMAINS if d not in state.get("domains_covered", [])
        ]
        if not pending:
            return

        next_domain = pending[0]
        language = state.get("language", "Español")
        messages = self._build_messages(state["transcript"], next_domain, language)
        yield from self._generate_stream(messages)

    def act_rapport(self, state: dict) -> dict:
        """Generates the therapist's turn during the rapport phase.

        Resolves (or randomly assigns) the patient name on the first turn,
        persists any assigned name back into ``client_profile``, and uses a
        rapport-specific system prompt with a turn-appropriate instruction.

        Args:
            state: Current ``SessionState``.

        Returns:
            Partial state dict with the updated transcript (and optionally an
            updated ``client_profile`` with the assigned name).
        """
        language = state.get("language", "Español")
        rapport_turns = state.get("rapport_turns", 0)

        # Resolve patient name; persist if randomly assigned so every agent
        # in the session uses the same name from this point on.
        patient_name, was_assigned = _resolve_patient_name(state, language)
        if was_assigned:
            profile = state.get("client_profile", {})
            profile.setdefault("demographics", {})["name"] = patient_name
            state["client_profile"] = profile

        messages = self._build_rapport_messages(
            state["transcript"], rapport_turns, language, patient_name
        )
        rapport_system = get_rapport_prompt(language)
        response = self._generate(messages, system_prompt=rapport_system)

        state["transcript"].append(
            {
                "role": "therapist",
                "content": response,
                "phase": "rapport",
                "turn_id": len(state["transcript"]),
            }
        )
        return state

    def _build_rapport_messages(
        self,
        transcript: list[dict],
        rapport_turns: int,
        language: str,
        patient_name: str = "",
    ) -> list[dict]:
        """Builds the message payload for a rapport-phase turn.

        The patient name is injected explicitly into the Turn-0 instruction so
        the model greets the patient by their real name instead of generating a
        placeholder like "[Nombre del Paciente]".  Each instruction is phrased
        as a concrete directive (not a meta-comment) so nothing leaks into the
        model's output.

        Args:
            transcript:    Full conversation so far.
            rapport_turns: How many rapport turns have already been completed.
            language:      Session language.
            patient_name:  Resolved patient name to use in the greeting.

        Returns:
            List of chat-completion message dicts.
        """
        messages: list[dict] = []
        for turn in transcript:
            role = "assistant" if turn["role"] == "therapist" else "user"
            messages.append({"role": role, "content": turn["content"]})

        name = patient_name or ("el paciente" if language == "Español" else "the patient")

        if language == "Español":
            if rapport_turns == 0:
                instruction = (
                    f"El paciente se llama {name}. "
                    f"Salúdale de forma cálida y natural usando su nombre "
                    f"(ejemplos válidos: 'Hola, {name}', '¡Buenas, {name}!', 'Hola {name}, qué tal'). "
                    "En no más de 3 frases di que el espacio es confidencial y pregunta "
                    "qué le trae hoy con una pregunta abierta. "
                    "Escribe SOLO lo que diría el terapeuta, sin acotaciones ni paréntesis."
                )
            elif rapport_turns == 1:
                instruction = (
                    "El paciente acaba de responder. "
                    "Primero refleja o valida brevemente lo que ha dicho para mostrar que le escuchas. "
                    "Luego haz UNA sola pregunta abierta para explorar más su motivo de consulta. "
                    "Escribe SOLO lo que diría el terapeuta, sin acotaciones ni paréntesis."
                )
            else:
                instruction = (
                    "Cierra la fase de apertura. "
                    "Resume en una frase lo que has escuchado, verifica con el paciente "
                    "('¿lo entendí bien?' o similar), y anuncia brevemente que vas a explorar "
                    "diferentes áreas para entender mejor su situación. "
                    "Escribe SOLO lo que diría el terapeuta, sin acotaciones ni paréntesis."
                )
        else:
            if rapport_turns == 0:
                instruction = (
                    f"The patient's name is {name}. "
                    f"Greet them warmly and naturally using their name "
                    f"(valid examples: 'Hi {name}', 'Hey {name}, good to meet you', 'Hello {name}'). "
                    "In no more than 3 sentences mention that this is a confidential space and ask "
                    "one open question about what brings them here today. "
                    "Write ONLY what the therapist would say, no stage directions or parentheses."
                )
            elif rapport_turns == 1:
                instruction = (
                    "The patient has just responded. "
                    "First briefly reflect or validate what they said to show you heard them. "
                    "Then ask ONE open question to explore their reason for coming further. "
                    "Write ONLY what the therapist would say, no stage directions or parentheses."
                )
            else:
                instruction = (
                    "Close the opening phase. "
                    "Summarise in one sentence what you have heard, verify with the patient "
                    "('Did I get that right?' or similar), and briefly announce that you will "
                    "now explore different areas to understand their situation better. "
                    "Write ONLY what the therapist would say, no stage directions or parentheses."
                )

        messages.append({"role": "user", "content": instruction})
        return messages

    def _build_messages(
        self, transcript: list[dict], target_domain: str, language: str
    ) -> list[dict]:
        """Constructs the message payload for the LLM."""
        messages = []
        for turn in transcript:
            role = "assistant" if turn["role"] == "therapist" else "user"
            messages.append({"role": role, "content": turn["content"]})

        # Add instruction for current turn
        instruction = f"Now, gently explore the following domain: {target_domain}. Ask ONE open-ended question. Reply in {language}."
        messages.append(
            {"role": "user", "content": instruction}
        )  # As system prompt injected by user role or system depending on model strategy

        return messages
