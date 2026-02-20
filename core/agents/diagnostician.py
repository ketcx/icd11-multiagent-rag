"""Diagnostician agent responsible for formulating ICD-11 hypotheses."""

from __future__ import annotations

import json
import re

from core.agents.base import BaseAgent

# Matches trailing commas before ] or } — a common LLM JSON defect.
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _sanitise_json(raw: str) -> str:
    """Strips trailing commas and normalises common LLM JSON defects.

    Models frequently emit constructs such as ``[..., ]`` or ``{..., }``
    which are invalid JSON but trivially fixable.  This function applies
    lightweight repairs before handing the string to ``json.loads``.

    Args:
        raw: Raw string extracted from the model response.

    Returns:
        Sanitised JSON string ready for ``json.loads``.
    """
    # Remove trailing commas before closing brackets/braces
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", raw)
    return cleaned


def _extract_json_block(response: str) -> str:
    """Extracts the JSON payload from a model response.

    Handles three common LLM output patterns:
    1. Fenced code block: ```json … ```
    2. Plain code block:  ``` … ```
    3. Raw JSON (no fencing)

    Args:
        response: Full model output string.

    Returns:
        Extracted JSON string (may still contain LLM defects; pass through
        ``_sanitise_json`` before parsing).
    """
    if "```json" in response:
        return response.split("```json")[1].split("```")[0].strip()
    if "```" in response:
        return response.split("```")[1].split("```")[0].strip()
    return response.strip()


class DiagnosticianAgent(BaseAgent):
    """Generates diagnostic hypotheses grounded in RAG evidence.

    Produces a JSON array of ``{label, code, confidence, evidence_for,
    evidence_against}`` objects.  Parsing is resilient to common LLM defects
    such as trailing commas, single-object responses, and markdown fencing.
    """

    def act(self, state: dict) -> dict:
        """Analyses the transcript and RAG chunks to formulate hypotheses.

        Args:
            state: Current ``SessionState`` with ``transcript`` and
                ``retrieved_chunks``.

        Returns:
            Partial state dict with ``hypotheses`` populated.
        """
        language = state.get("language", "Español")
        messages = self._build_messages(
            state["transcript"],
            state.get("retrieved_chunks", []),
            language,
        )
        response = self._generate(messages)

        hypotheses: list[dict] = []
        try:
            json_str = _sanitise_json(_extract_json_block(response))
            parsed = json.loads(json_str)
            hypotheses = parsed if isinstance(parsed, list) else [parsed]
        except Exception as exc:
            # Surface the raw LLM output as a single low-confidence entry so
            # the session can continue and the user can inspect the raw text.
            hypotheses = [
                {
                    "label": "JSON parse error — raw model output attached",
                    "code": "N/A",
                    "confidence": "LOW",
                    "evidence_for": [response],
                    "evidence_against": [str(exc)],
                }
            ]

        state["hypotheses"] = hypotheses
        return state

    def _build_messages(
        self,
        transcript: list[dict],
        chunks: list[dict],
        language: str,
    ) -> list[dict]:
        """Injects the ICD-11 context and transcript into the prompt payload.

        Args:
            transcript: Full conversation transcript.
            chunks:     Retrieved ICD-11 context chunks.
            language:   Session language for the response instruction.

        Returns:
            List of chat-completion message dicts.
        """
        context_str = (
            "\n---\n".join(c["content"] for c in chunks) if chunks else "No context available."
        )
        transcript_str = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in transcript
        )

        prompt = (
            f"TRANSCRIPT:\n{transcript_str}\n\n"
            f"ICD-11 CONTEXT:\n{context_str}\n\n"
            f"Analyse the transcript and generate diagnostic hypotheses grounded ONLY in the "
            f"provided ICD-11 context. Reply in {language}. "
            f"Output ONLY a valid JSON array with no trailing commas. Do NOT wrap it in markdown."
        )

        return [{"role": "user", "content": prompt}]
