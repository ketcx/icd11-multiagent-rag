"""Base agent template with local LLM integration (llama-cpp-python).

``llama-cpp-python`` is imported lazily inside ``_generate`` so that the
module loads cleanly even when the package is not installed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_MAX_RETRIES = 1  # One automatic retry on transient LLM failure


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
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def act(self, state: dict) -> dict:
        """Executes the agent's action and returns the updated state.

        Args:
            state: The current ``SessionState`` dictionary.

        Returns:
            A (partial) dict with the keys this agent modifies.
        """

    def _generate(self, messages: list[dict]) -> str:
        """Generates a response using the local LLM with retry logic.

        Prepends the system prompt as a ``system`` role message, attempts
        inference up to ``_MAX_RETRIES + 1`` times, and returns an empty
        string on persistent failure rather than propagating the exception.

        Args:
            messages: Chat-completion messages (user/assistant turns).

        Returns:
            The model's generated text, or ``""`` if inference fails after all
            retries.
        """
        if self.llm is None:
            return ""

        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.llm.create_chat_completion(  # type: ignore[union-attr]
                    messages=full_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response["choices"][0]["message"]["content"]  # type: ignore[index]
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
