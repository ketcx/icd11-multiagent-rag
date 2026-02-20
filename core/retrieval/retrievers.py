"""Retrievers: dense (Chroma) + lexical (BM25) with fusion."""

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Combines dense retrieval (Chroma) + BM25 with Reciprocal Rank Fusion."""

    def __init__(self, vectorstore: Chroma, bm25_corpus: list[dict],
                 top_k_dense: int = 8, top_k_bm25: int = 8,
                 top_k_final: int = 6):
        self.vectorstore = vectorstore
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final
        # Build BM25 index
        tokenized = [doc["content"].lower().split() for doc in bm25_corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_docs = bm25_corpus

    def _rrf_fusion(self, dense_results: list, bm25_top: list) -> list[dict]:
        """Merges dense and BM25 ranked lists via Reciprocal Rank Fusion.

        Each document is scored as the sum of 1/(k + rank) across both ranked
        lists, where k=60 is the RRF smoothing constant that dampens the
        influence of very high-ranked documents.  Documents appearing in both
        lists receive additive score contributions, naturally surfacing items
        that are strong in multiple retrieval dimensions.

        Args:
            dense_results: List of (Document, float) tuples from Chroma.
            bm25_top: List of (corpus_index, bm25_score) tuples, pre-sorted
                descending by BM25 score.

        Returns:
            Up to ``self.top_k_final`` deduplicated chunks sorted by fused
            score descending, each as ``{content, metadata, score, source}``.
        """
        K = 60  # Standard RRF smoothing constant
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        # Contribution from dense retrieval
        for rank, (doc, _similarity) in enumerate(dense_results):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank + 1)
            if key not in docs:
                docs[key] = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": "dense",
                }

        # Contribution from BM25 retrieval
        for rank, (corpus_idx, _bm25_score) in enumerate(bm25_top):
            if corpus_idx >= len(self.bm25_docs):
                continue
            bm25_doc = self.bm25_docs[corpus_idx]
            key = bm25_doc["content"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank + 1)
            if key not in docs:
                docs[key] = {
                    "content": bm25_doc["content"],
                    "metadata": bm25_doc.get("metadata", {}),
                    "source": "bm25",
                }
            else:
                # Document appeared in both lists
                docs[key]["source"] = "hybrid"

        # Sort by fused RRF score and attach score to each result
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Prioritise chunks that contain an explicit ICD-11 code
        def _priority(item: tuple[str, float]) -> tuple[int, float]:
            key, score = item
            has_code = bool(docs[key].get("metadata", {}).get("code"))
            return (0 if has_code else 1, -score)

        ranked.sort(key=_priority)

        result = []
        for key, score in ranked[: self.top_k_final]:
            entry = dict(docs[key])
            entry["score"] = round(score, 6)
            result.append(entry)

        return result

    def retrieve(self, query: str, filter_metadata: dict | None = None) -> list[dict]:
        """Executes hybrid retrieval.

        1. Dense search on Chroma
        2. BM25 search
        3. Reciprocal Rank Fusion
        4. Dedup + prioritize chunks with code/uri
        """
        # Dense
        dense_results = self.vectorstore.similarity_search_with_score(
            query, k=self.top_k_dense, filter=filter_metadata
        )
        # BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )[:self.top_k_bm25]

        # Fusion (Reciprocal Rank Fusion, k=60)
        return self._rrf_fusion(dense_results, bm25_top)
