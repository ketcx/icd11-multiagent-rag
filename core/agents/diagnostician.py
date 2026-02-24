"""Diagnostician agent responsible for formulating ICD-11 hypotheses."""

from __future__ import annotations

import json
import re

from core.agents.base import BaseAgent

# Matches trailing commas before ] or } — a common LLM JSON defect.
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _sanitise_json(raw: str) -> str:
    """Strips trailing commas and normalises common LLM JSON defects."""
    return _TRAILING_COMMA_RE.sub(r"\1", raw)


def _extract_json_block(response: str) -> str:
    """Extracts the JSON payload from a model response.

    Strategy (in order):
    1. Fenced code block — ````json … ``` ``
    2. Plain code block  — ```` ``` … ``` ``
    3. Bracket-aware scan — finds the first ``[`` or ``{`` and walks the
       string tracking depth and string literals to locate the matching
       closing bracket.  This correctly handles nested structures and stops
       before any hallucinated/looping text that follows the JSON.

    Returns the extracted candidate string; pass through ``_sanitise_json``
    before calling ``json.loads``.
    """
    if "```json" in response:
        return response.split("```json")[1].split("```")[0].strip()
    if "```" in response:
        return response.split("```")[1].split("```")[0].strip()

    # Pick whichever bracket type appears first in the response so that a
    # single JSON object (``{…}``) whose values contain ``[]`` is not
    # mistakenly matched by the ``[`` scanner before the ``{`` scanner.
    candidates = []
    for pair in [("[", "]"), ("{", "}")]:
        idx = response.find(pair[0])
        if idx != -1:
            candidates.append((idx, pair))
    candidates.sort(key=lambda c: c[0])

    for _, (start_char, end_char) in candidates:
        start_idx = response.find(start_char)
        if start_idx == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(response[start_idx:], start=start_idx):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return response[start_idx : i + 1]
        # Closing bracket not found (truncated output) — return what we have.
        return response[start_idx:].strip()

    return response.strip()


def _extract_partial_hypotheses(response: str) -> list[dict]:
    """Fallback: extracts every complete JSON object using bracket-walking.

    Handles nested structures (arrays inside objects) correctly, unlike a
    simple non-nested regex.  Used when the model truncates mid-array so that
    valid objects produced before truncation are preserved.
    """
    results = []
    i = 0
    n = len(response)
    while i < n:
        if response[i] != "{":
            i += 1
            continue
        # Walk brackets to find the matching closing brace.
        depth = 0
        in_string = False
        escape = False
        end = -1
        for j in range(i, n):
            ch = response[j]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            break  # Truncated — no more complete objects.
        obj_str = response[i : end + 1]
        try:
            obj = json.loads(_sanitise_json(obj_str))
            if isinstance(obj, dict) and "label" in obj:
                results.append(obj)
        except Exception:
            pass
        i = end + 1
    return results


def _deduplicate_hypotheses(hypotheses: list[dict]) -> list[dict]:
    """Removes duplicate hypotheses and deduplicates evidence lists.

    Two hypotheses are considered duplicates when they share the same
    ``label`` (case-insensitive) or ``code``.  When duplicates exist the
    first occurrence is kept and its evidence lists are merged (deduplicated).
    Within each hypothesis, ``evidence_for`` and ``evidence_against`` entries
    are also deduplicated (case-insensitive, order-preserved).
    """

    def _dedup_list(items: list) -> list:
        """Preserves order, removes case-insensitive duplicates."""
        seen: set[str] = set()
        result: list = []
        for item in items:
            key = str(item).strip().lower()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    seen_labels: dict[str, int] = {}  # normalised label → index in result
    seen_codes: dict[str, int] = {}  # normalised code  → index in result
    result: list[dict] = []

    for h in hypotheses:
        label_key = str(h.get("label", "")).strip().lower()
        code_key = str(h.get("code", "")).strip().upper()

        existing_idx = seen_labels.get(label_key) or seen_codes.get(code_key)

        if existing_idx is not None:
            # Merge evidence into the first occurrence
            existing = result[existing_idx]
            existing["evidence_for"] = _dedup_list(
                existing.get("evidence_for", []) + h.get("evidence_for", [])
            )
            existing["evidence_against"] = _dedup_list(
                existing.get("evidence_against", []) + h.get("evidence_against", [])
            )
        else:
            # Deduplicate the evidence lists of this hypothesis too
            h["evidence_for"] = _dedup_list(h.get("evidence_for", []))
            h["evidence_against"] = _dedup_list(h.get("evidence_against", []))
            idx = len(result)
            result.append(h)
            if label_key:
                seen_labels[label_key] = idx
            if code_key:
                seen_codes[code_key] = idx

    return result


def _is_valid_hypothesis(h: dict) -> bool:
    """Returns True only if the hypothesis has a meaningful label and code."""
    label = str(h.get("label", "")).strip().strip("`\"'")
    code = str(h.get("code", "")).strip().strip("`\"'")
    # Must have a non-empty label that isn't a placeholder / parse-error marker
    if not label or label.lower() in ("", "n/a", "unknown", "none", "null"):
        return False
    # Must have a code that looks like a real ICD-11 code (letter + digits pattern)
    return bool(code) and code.upper() not in ("", "N/A", "UNKNOWN", "NONE", "NULL")


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
        from core.agents.prompts import get_diagnostician_prompt

        language = state.get("language", "Español")
        messages = self._build_messages(
            state["transcript"],
            state.get("retrieved_chunks", []),
            language,
        )
        response = self._generate(
            messages,
            system_prompt=get_diagnostician_prompt(language),
            postprocess=False,
        )

        hypotheses: list[dict] = []
        try:
            json_str = _sanitise_json(_extract_json_block(response))
            parsed = json.loads(json_str)
            hypotheses = parsed if isinstance(parsed, list) else [parsed]
        except Exception as exc:
            # Try to salvage any complete objects produced before the model
            # went off-rails (e.g. mid-array hallucination).
            hypotheses = _extract_partial_hypotheses(response)
            if not hypotheses:
                hypotheses = [
                    {
                        "label": "Error de análisis — salida del modelo adjunta",
                        "code": "ERROR",
                        "confidence": "LOW",
                        "evidence_for": [response[:500]],
                        "evidence_against": [str(exc)],
                    }
                ]

        state["hypotheses"] = [
            h for h in _deduplicate_hypotheses(hypotheses) if _is_valid_hypothesis(h)
        ]
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

        if language == "Español":
            prompt = (
                f"TRANSCRIPCIÓN:\n{transcript_str}\n\n"
                f"CONTEXTO CIE-11:\n{context_str}\n\n"
                f"Analiza la transcripción y genera hipótesis diagnósticas basadas ÚNICAMENTE "
                f"en el contexto CIE-11 proporcionado. Responde COMPLETAMENTE en español.\n\n"
                f"REGLAS CRÍTICAS DE SALIDA:\n"
                f"1. Escribe ÚNICAMENTE un array JSON válido. Sin texto antes ni después.\n"
                f"2. NO uses bloques de código markdown.\n"
                f"3. Sin comas al final.\n"
                f"4. DETENTE inmediatamente después del corchete ] de cierre.\n"
                f"5. Cada objeto debe tener exactamente estas claves: "
                f'"label", "code", "confidence", "evidence_for", "evidence_against".\n'
                f"6. evidence_for y evidence_against deben ser listas de frases CORTAS "
                f"en español (máx. 10 palabras cada una). NO copies frases completas de la transcripción.\n"
                f"7. Máximo 3 elementos por lista de evidencia.\n\n"
                f"Salida JSON:\n"
            )
        else:
            prompt = (
                f"TRANSCRIPT:\n{transcript_str}\n\n"
                f"ICD-11 CONTEXT:\n{context_str}\n\n"
                f"Analyse the transcript and generate diagnostic hypotheses grounded ONLY in the "
                f"provided ICD-11 context. Reply entirely in English.\n\n"
                f"CRITICAL OUTPUT RULES:\n"
                f"1. Output ONLY a valid JSON array. No text before or after the JSON.\n"
                f"2. Do NOT wrap it in markdown code fences.\n"
                f"3. No trailing commas.\n"
                f"4. STOP immediately after the closing ] bracket.\n"
                f"5. Each object must have exactly these keys: "
                f'"label", "code", "confidence", "evidence_for", "evidence_against".\n'
                f"6. evidence_for and evidence_against must be lists of SHORT phrases "
                f"in English (max 10 words each). Do NOT copy full sentences from the transcript.\n"
                f"7. Maximum 3 items per evidence list.\n\n"
                f"JSON Output:\n"
            )

        return [{"role": "user", "content": prompt}]
