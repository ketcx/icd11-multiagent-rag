"""Therapist agent responsible for interviewing the client."""

from core.agents.base import BaseAgent

class TherapistAgent(BaseAgent):
    """Explores clinical domains empathetically without diagnosing."""
    
    # Target domains to evaluate
    DOMAINS = [
        "mood", "anxiety", "sleep", "eating", "substances",
        "psychosis", "trauma", "ocd", "cognition",
        "social_functioning", "suicidal_ideation"
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
        state["transcript"].append({
            "role": "therapist",
            "content": response,
            "domain": next_domain,
            "turn_id": len(state["transcript"]),
        })
        
        # Explicitly set the pending domains for external tracking
        state["domains_pending"] = pending
        
        return state

    def _build_messages(self, transcript: list[dict], target_domain: str, language: str) -> list[dict]:
        """Constructs the message payload for the LLM."""
        messages = []
        for turn in transcript:
            role = "assistant" if turn["role"] == "therapist" else "user"
            messages.append({"role": role, "content": turn["content"]})
            
        # Add instruction for current turn
        instruction = f"Now, gently explore the following domain: {target_domain}. Ask ONE open-ended question. Reply in {language}."
        messages.append({"role": "user", "content": instruction}) # As system prompt injected by user role or system depending on model strategy
        
        return messages
