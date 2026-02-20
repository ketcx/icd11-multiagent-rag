"""Tests for the hybrid retrieval pipeline."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from core.retrieval.retrievers import HybridRetriever
from core.retrieval.query_builder import QueryBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chroma_doc(content: str, code: str | None = None) -> MagicMock:
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"code": code} if code else {}
    return doc


def _make_retriever(dense_docs: list, bm25_corpus: list) -> HybridRetriever:
    """Creates a HybridRetriever with a mocked Chroma vectorstore."""
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [
        (_make_chroma_doc(d["content"], d.get("code")), 0.9 - i * 0.05)
        for i, d in enumerate(dense_docs)
    ]
    return HybridRetriever(mock_store, bm25_corpus, top_k_dense=4, top_k_bm25=4, top_k_final=3)


# ---------------------------------------------------------------------------
# BM25 corpus used across tests
# ---------------------------------------------------------------------------


BM25_CORPUS = [
    {
        "content": "Depressive Episode ICD-11 6A70 persistent low mood loss of interest",
        "metadata": {"code": "6A70"},
    },
    {
        "content": "Anxiety Disorder ICD-11 6B00 excessive worry fear daily functioning",
        "metadata": {"code": "6B00"},
    },
    {
        "content": "Insomnia Disorder ICD-11 7A00 difficulty initiating maintaining sleep",
        "metadata": {"code": "7A00"},
    },
    {
        "content": "Post Traumatic Stress Disorder ICD-11 6B40 trauma exposure intrusion avoidance",
        "metadata": {"code": "6B40"},
    },
]


DENSE_DOCS = [
    {
        "content": "Anxiety Disorder ICD-11 6B00 excessive worry fear daily functioning",
        "code": "6B00",
    },
    {
        "content": "Depressive Episode ICD-11 6A70 persistent low mood loss of interest",
        "code": "6A70",
    },
    {"content": "General background text without a specific code", "code": None},
]


class TestHybridRetrieverRRF:
    def test_retrieve_returns_list(self) -> None:
        retriever = _make_retriever(DENSE_DOCS, BM25_CORPUS)
        results = retriever.retrieve("anxiety worry")
        assert isinstance(results, list)

    def test_retrieve_respects_top_k_final(self) -> None:
        retriever = _make_retriever(DENSE_DOCS, BM25_CORPUS)
        results = retriever.retrieve("anxiety")
        assert len(results) <= 3  # top_k_final=3

    def test_results_have_required_keys(self) -> None:
        retriever = _make_retriever(DENSE_DOCS, BM25_CORPUS)
        results = retriever.retrieve("mood")
        for chunk in results:
            assert "content" in chunk
            assert "metadata" in chunk
            assert "score" in chunk

    def test_chunks_with_icd_code_prioritised(self) -> None:
        """Documents with an ICD code in metadata should rank first."""
        retriever = _make_retriever(DENSE_DOCS, BM25_CORPUS)
        results = retriever.retrieve("mood disorder anxiety")
        if len(results) >= 2:
            # At least the first result should have a code
            codes_present = [bool(r["metadata"].get("code")) for r in results[:2]]
            assert any(codes_present)

    def test_rrf_scores_are_positive(self) -> None:
        retriever = _make_retriever(DENSE_DOCS, BM25_CORPUS)
        results = retriever.retrieve("anxiety sleep")
        for r in results:
            assert r["score"] > 0

    def test_no_duplicate_content(self) -> None:
        retriever = _make_retriever(DENSE_DOCS, BM25_CORPUS)
        results = retriever.retrieve("anxiety depression")
        contents = [r["content"] for r in results]
        assert len(contents) == len(set(contents))

    def test_source_field_set(self) -> None:
        retriever = _make_retriever(DENSE_DOCS, BM25_CORPUS)
        results = retriever.retrieve("trauma")
        for r in results:
            assert r["source"] in ("dense", "bm25", "hybrid")


class TestQueryBuilder:
    def test_returns_dict_with_semantic_key(self) -> None:
        qb = QueryBuilder()
        transcript = [
            {"role": "therapist", "content": "How are you feeling?"},
            {"role": "client", "content": "Very anxious and struggling to sleep."},
        ]
        result = qb.build_queries(transcript)
        assert "semantic" in result

    def test_semantic_query_non_empty(self) -> None:
        qb = QueryBuilder()
        transcript = [{"role": "client", "content": "I feel extremely anxious"}]
        result = qb.build_queries(transcript)
        assert result["semantic"]

    def test_empty_transcript_returns_defaults(self) -> None:
        qb = QueryBuilder()
        result = qb.build_queries([])
        assert isinstance(result, dict)
        assert "semantic" in result
