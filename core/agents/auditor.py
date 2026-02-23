"""Evidence Auditor agent for internal fact-checking."""

from __future__ import annotations

import re

from core.agents.base import BaseAgent


class EvidenceAuditorAgent(BaseAgent):
    """Verifies traceability and factual grounding of diagnostic hypotheses.

    For each hypothesis the auditor checks that every ``evidence_for`` claim
    can be traced back to either:

    - A turn in the session transcript (substring or keyword match), or
    - A retrieved ICD-11 chunk (substring or keyword match).

    Unverifiable claims are surfaced as issues in the audit report.  A
    ``traceability_score`` in [0, 1] is computed as the fraction of claims
    that could be grounded.

    When the LLM is available the auditor also prompts the model for a
    free-text commentary on the overall evidential quality.
    """

    # Minimum token length to consider a claim word meaningful
    _MIN_TOKEN_LEN = 4

    def act(self, state: dict) -> dict:
        """Audits hypotheses and returns an enriched ``audit_report``.

        Args:
            state: Current ``SessionState`` containing ``hypotheses``,
                ``transcript``, and ``retrieved_chunks``.

        Returns:
            State dict with ``audit_report`` key populated.
        """
        hypotheses: list[dict] = state.get("hypotheses", [])
        transcript: list[dict] = state.get("transcript", [])
        chunks: list[dict] = state.get("retrieved_chunks", [])

        transcript_text = " ".join(t.get("content", "") for t in transcript).lower()
        chunk_text = " ".join(c.get("content", "") for c in chunks).lower()

        all_issues: list[dict] = []
        total_claims = 0
        grounded_claims = 0

        for hypothesis in hypotheses:
            label = hypothesis.get("label", "Unknown")
            evidence_for: list = hypothesis.get("evidence_for", [])

            for claim in evidence_for:
                claim_str = str(claim).lower().strip()
                total_claims += 1
                in_transcript = self._is_grounded(claim_str, transcript_text)
                in_chunks = self._is_grounded(claim_str, chunk_text)

                if in_transcript or in_chunks:
                    grounded_claims += 1
                else:
                    all_issues.append(
                        {
                            "hypothesis": label,
                            "claim": str(claim),
                            "reason": "No matching evidence found in transcript or retrieved chunks.",
                        }
                    )

        traceability_score = round(grounded_claims / total_claims, 4) if total_claims > 0 else 1.0

        llm_commentary: str | None = None
        if self.llm is not None and hypotheses:
            llm_commentary = self._request_llm_commentary(state, all_issues)

        report = {
            "verified": len(all_issues) == 0,
            "traceability_score": traceability_score,
            "total_claims": total_claims,
            "grounded_claims": grounded_claims,
            "issues": all_issues,
            "llm_commentary": llm_commentary,
        }

        state["audit_report"] = report
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_grounded(self, claim: str, corpus: str) -> bool:
        """Returns True if any meaningful token from *claim* appears in *corpus*."""
        tokens = [t for t in re.split(r"\W+", claim) if len(t) >= self._MIN_TOKEN_LEN]
        if not tokens:
            return True  # Cannot verify an empty claim; assume grounded
        return any(token in corpus for token in tokens)

    def _request_llm_commentary(self, state: dict, issues: list[dict]) -> str | None:
        """Asks the LLM to comment on evidence quality; returns None on failure."""
        language = state.get("language", "Español")
        hypotheses_summary = "; ".join(h.get("label", "?") for h in state.get("hypotheses", []))
        unverified_count = len(issues)

        if language == "Español":
            prompt = (
                f"Revisa la calidad evidencial de las siguientes hipótesis diagnósticas: "
                f"{hypotheses_summary}. Se encontraron {unverified_count} afirmaciones sin "
                f"respaldo en la transcripción o contexto recuperado. "
                f"Proporciona un comentario conciso (máximo 3 oraciones) sobre la solidez "
                f"del respaldo evidencial."
            )
        else:
            prompt = (
                f"Review the evidential quality of the following diagnostic hypotheses: "
                f"{hypotheses_summary}. There are {unverified_count} claims not grounded in "
                f"the transcript or retrieved context. "
                f"Provide a concise commentary (max 3 sentences) on the strength of evidential support."
            )

        messages = [{"role": "user", "content": prompt}]
        return self._generate(messages)
