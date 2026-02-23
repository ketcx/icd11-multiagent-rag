"""Constructs optimized queries based on the conversation state."""


class QueryBuilder:
    """Extracts clinical entities from the transcript and generates retrieval queries.

    Generates 2 types of queries:
    (a) Semantic: symptom summary + clinical context
    (b) Exact-match: ICD-11 codes, exact diagnostic terms
    """

    def build_queries(
        self,
        transcript: list[dict],
        conversation_summary: str = "",
        identified_symptoms: list[str] | None = None,
    ) -> dict:
        """
        Returns:
            {
                "semantic": "paciente presenta estado de ánimo deprimido...",
                "exact": ["6A70", "trastorno depresivo", "episodio depresivo"]
            }
        """
        if identified_symptoms is None:
            identified_symptoms = []

        # Simple extraction strategy for now
        last_turn = transcript[-1]["content"] if transcript else ""
        return {"semantic": str(last_turn), "exact": []}
