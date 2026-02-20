"""Builds/Updates the ChromaDB index from chunks."""

from langchain_chroma import Chroma
from langchain.schema import Document

COLLECTION_NAME = "icd11_es"
PERSIST_DIR = "data/indexes/chroma"


def build_index(chunks: list[dict], embedding_model_name: str, device: str = "mps") -> Chroma:
    """Builds ChromaDB from processed chunks.

    Args:
        chunks: List of dicts {content, metadata}
        embedding_model_name: "NeuML/pubmedbert-base-embeddings"
        device: "mps" or "cpu"

    Returns:
        Chroma vectorstore ready for queries
    """
    # Initialize embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

    # Create LangChain documents
    documents = [Document(page_content=c["content"], metadata=c["metadata"] or {}) for c in chunks]

    # Build and persist
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore
