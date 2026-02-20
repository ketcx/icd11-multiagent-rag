"""RAG retrieval pipeline initialization."""

from core.retrieval.query_builder import QueryBuilder
from core.retrieval.retrievers import HybridRetriever

__all__ = ["QueryBuilder", "HybridRetriever", "init_rag_pipeline"]

# Global RAG pipeline state
_rag_pipeline = None


class SimpleEmbeddings:
    """Ultra-lightweight embeddings using TF-IDF-like vectorization."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create sparse TF-IDF-like vectors."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
            vectors = vectorizer.fit_transform(texts).toarray()
            return vectors.tolist()
        except Exception as e:
            # Fallback: return zeros
            print(f"⚠️  TF-IDF embedding failed: {e}. Using zeros.")
            return [[0.0] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        result = self.embed_documents([text])
        return result[0]


def init_rag_pipeline():
    """Lazily initializes the RAG pipeline (Chroma vectorstore + HybridRetriever)."""
    global _rag_pipeline

    if _rag_pipeline is not None:
        return _rag_pipeline

    try:
        from pathlib import Path

        from langchain_chroma import Chroma

        # Path to the pre-built Chroma index
        chroma_path = Path(__file__).parent.parent.parent / "data" / "indexes" / "chroma"

        if not chroma_path.exists() or not any(chroma_path.glob("*")):
            print(f"⚠️ Chroma index not found at {chroma_path}. RAG will use mock context.")
            return None

        # Load collection name from config (fall back to default)
        try:
            import yaml

            cfg_path = Path(__file__).parent.parent.parent / "configs" / "app.yaml"
            with open(cfg_path) as fh:
                _cfg = yaml.safe_load(fh)
            collection_name: str = _cfg.get("retrieval", {}).get("collection_name", "icd11_es")
        except Exception:
            collection_name = "icd11_es"

        # Load the pre-built Chroma vectorstore with lightweight embeddings
        embeddings = SimpleEmbeddings()
        vectorstore = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embeddings,
            collection_name=collection_name,
        )

        # Initialize HybridRetriever with mock BM25 corpus
        bm25_corpus = [
            {
                "content": "Depressive Episode - ICD-11 6A70: A depressive episode is characterized by persistent low mood and loss of interest in activities.",
                "metadata": {"code": "6A70"},
            },
            {
                "content": "Anxiety Disorder - ICD-11 6A80: Anxiety is excessive worry and fear that interferes with daily functioning.",
                "metadata": {"code": "6A80"},
            },
        ]

        retriever = HybridRetriever(vectorstore, bm25_corpus)
        _rag_pipeline = {
            "vectorstore": vectorstore,
            "retriever": retriever,
            "query_builder": QueryBuilder(),
        }

        print("✓ RAG pipeline initialized successfully")
        return _rag_pipeline

    except Exception as e:
        print(f"⚠️ Failed to initialize RAG pipeline: {e}")
        return None


def get_rag_pipeline():
    """Retrieves the initialized RAG pipeline, or initializes it if needed."""
    return init_rag_pipeline()
