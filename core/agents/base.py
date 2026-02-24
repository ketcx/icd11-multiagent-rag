"""Base agent template with local LLM integration (llama-cpp-python).

``llama-cpp-python`` is imported lazily inside ``_generate`` so that the
module loads cleanly even when the package is not installed.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_MAX_RETRIES = 1  # One automatic retry on transient LLM failure

# Stop tokens for Phi-3 / ChatML format.  Without these the model overshoots
# its answer and starts generating the next conversation turn's template
# markers (e.g. "<|user|>", "<|im_start|>system …").
# "<|" catches any other Phi-3 special-token leak (e.g. "<|emotions", "<|emotional").
_CHATML_STOP = ["<|end|>", "<|user|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|"]

# Parenthetical meta-comments the model sometimes appends to its output,
# e.g. "(Asegúrate de escuchar activamente…)" or "(Make sure to adjust tone…)".
# These are stage directions, not therapist dialogue, and must be removed.
_STAGE_DIRECTION_RE = re.compile(r"\s*\([^)]{20,}\)\s*", re.DOTALL)

# Role-label artifacts the model sometimes generates after real dialogue.
# We match ONLY at the start of the string or after a newline, and require a
# colon/angle-bracket separator, so mid-sentence uses of "user" or "patient"
# in normal clinical text are NOT affected.
# Also catches "roleplaying" unconditionally since it is never valid dialogue.
_ROLE_ARTIFACT_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:user|model|usuario|modelo|assistant|human)\s*[:>|]"
    r"|(?:^|\n)\s*roleplaying",
    re.IGNORECASE,
)
# HTML/angle-bracket garbage: <response, <logical, <message, etc.
# Also catches Phi-3 special-token leaks: <|emotions, <|emotional, etc.
_TAG_GARBAGE_RE = re.compile(r"<[a-z|]", re.IGNORECASE)

# Degeneration / repetition-collapse detector.
# Matches when the same 4+ character token (all caps or mixed) repeats 3+ times in
# close proximity, e.g. "NOGPT: NOGPT: NOGPT" or "GATHER:NOGATHER:GATHER".
_DEGENERATION_RE = re.compile(
    r"(\b[A-Z][A-Z_:]{3,})(?:[:\s,;.]*\1){2,}",
    re.IGNORECASE,
)

# Catches sequences of ALL-CAPS "words" that aren't real (4+ chars, 3+ in a row).
# e.g., "NOOLOGY. NOGATOKEN. NOGPTHER:"
_ALLCAPS_GIBBERISH_RE = re.compile(
    r"(?:\b[A-Z]{4,}\b[\s:.,;]*){3,}",
)


def _strip_stage_directions(text: str) -> str:
    """Removes parenthetical stage directions injected by the model.

    Only removes parenthetical blocks that are at least 20 characters long
    (to avoid stripping genuine short parentheticals like "(0-10)" in clinical
    questions) and collapses any resulting double spaces.
    """
    cleaned = _STAGE_DIRECTION_RE.sub(" ", text)
    return " ".join(cleaned.split()).strip()


def _strip_degeneration(text: str) -> str:
    """Truncates output at the first sign of repetitive token collapse.

    Small quantised models occasionally degenerate into sequences like
    ``NOGPT: NOGPTHER: NOGATHER:GATHER:DATA`` or runs of ALL-CAPS gibberish.
    This function keeps only the coherent text before the collapse point.
    """
    # Check for repeated-token patterns
    m = _DEGENERATION_RE.search(text)
    if m:
        text = text[: m.start()]

    # Check for runs of ALL-CAPS gibberish words
    m = _ALLCAPS_GIBBERISH_RE.search(text)
    if m:
        text = text[: m.start()]

    return text.strip()


def _strip_role_artifacts(text: str) -> str:
    """Truncates output at the first role-label or HTML-garbage marker.

    When the model loses track of its role it starts outputting meta-text such
    as ``user:model:modeling session:`` or ``<response. message:user-sentatext``.
    This function keeps only the text that precedes the first such marker so
    that any valid patient/therapist dialogue produced before the model went
    off-rails is preserved.
    """
    # Truncate at the first HTML/angle-bracket garbage
    m_tag = _TAG_GARBAGE_RE.search(text)
    if m_tag:
        text = text[: m_tag.start()]

    # Truncate at the first role-label artifact
    m_role = _ROLE_ARTIFACT_RE.search(text)
    if m_role:
        text = text[: m_role.start()]

    return text.strip()


class BaseAgent(ABC):
    """Common interface for all agents within the system.

    Wraps a ``llama_cpp.Llama`` instance and provides a single ``_generate``
    method with retry logic and graceful fallback so that individual agents do
    not need to handle LLM errors themselves.

    Attributes:
        llm: The loaded ``Llama`` instance, or ``None`` when running in mock
            mode (e.g., during tests or when models are unavailable).
        system_prompt: The system-role message prepended to every inference
            call.
        temperature: Sampling temperature forwarded to the model.
        max_tokens: Maximum new tokens to generate per call.
    """

    def __init__(
        self,
        llm: object | None,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        repeat_penalty: float = 1.3,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty

    @abstractmethod
    def act(self, state: dict) -> dict:
        """Executes the agent's action and returns the updated state.

        Args:
            state: The current ``SessionState`` dictionary.

        Returns:
            A (partial) dict with the keys this agent modifies.
        """

    def _generate(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        *,
        postprocess: bool = True,
    ) -> str:
        """Generates a response using the local LLM with retry logic.

        Prepends the system prompt as a ``system`` role message, attempts
        inference up to ``_MAX_RETRIES + 1`` times, and returns an empty
        string on persistent failure rather than propagating the exception.

        Args:
            messages:      Chat-completion messages (user/assistant turns).
            system_prompt: Optional override for ``self.system_prompt``.  Used
                           by the rapport phase which needs a different system
                           instruction than the clinical exploration phase.
            postprocess:   When ``True`` (default), strips role-label artifacts
                           and stage directions from the output.  Set to
                           ``False`` for agents that produce structured output
                           (e.g. JSON) where such stripping would corrupt the
                           result.

        Returns:
            The model's generated text, or ``""`` if inference fails after all
            retries.
        """
        if self.llm is None:
            return ""

        sp = system_prompt if system_prompt is not None else self.system_prompt
        full_messages = [{"role": "system", "content": sp}] + messages

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.llm.create_chat_completion(  # type: ignore[union-attr, attr-defined]
                    messages=full_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=_CHATML_STOP,
                    repeat_penalty=self.repeat_penalty,
                )
                content: str = response["choices"][0]["message"]["content"]  # type: ignore[index]
                # Strip any ChatML markers that leaked through despite stop tokens.
                for token in _CHATML_STOP:
                    content = content.split(token)[0]
                if postprocess:
                    # Remove role-label artifacts (user:, model:, roleplaying …, <response …)
                    content = _strip_role_artifacts(content)
                    # Remove parenthetical stage directions the model sometimes appends.
                    content = _strip_stage_directions(content)
                    # Truncate at repetitive gibberish / degeneration collapse.
                    content = _strip_degeneration(content)
                return content
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "LLM inference failed (attempt %d/%d): %s — retrying.",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        exc,
                    )
                else:
                    logger.error(
                        "LLM inference failed after %d attempt(s): %s — returning empty string.",
                        _MAX_RETRIES + 1,
                        exc,
                    )

        return ""

    def _generate_stream(self, messages: list[dict], system_prompt: str | None = None):
        """Token generator for real-time streaming via llama-cpp-python.

        Prepends the system prompt and yields content delta strings one at a
        time as the model produces them.  Stops silently on any error so that
        callers do not need to handle exceptions.

        Args:
            messages:      Chat-completion messages (user/assistant turns).
            system_prompt: Optional override for ``self.system_prompt``.

        Yields:
            Non-empty string tokens from the model's streamed response.
        """
        if self.llm is None:
            return

        sp = system_prompt if system_prompt is not None else self.system_prompt
        full_messages = [{"role": "system", "content": sp}] + messages

        try:
            response = self.llm.create_chat_completion(  # type: ignore[union-attr, attr-defined]
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=_CHATML_STOP,
                repeat_penalty=self.repeat_penalty,
                stream=True,
            )
            for chunk in response:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("LLM streaming failed: %s", exc)
