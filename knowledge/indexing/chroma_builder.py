"""Builds/Updates the ChromaDB index from chunks."""

from __future__ import annotations

from pathlib import Path

from langchain.schema import Document
from langchain_chroma import Chroma

COLLECTION_NAME = "icd11_es"
PERSIST_DIR = "data/indexes/chroma"


def build_chroma_index(
    chunks: list[dict],
    embedding_model_name: str,
    device: str = "mps",
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Builds ChromaDB from processed chunks.

    Args:
        chunks: List of dicts {content, metadata}
        embedding_model_name: "NeuML/pubmedbert-base-embeddings"
        device: "mps" or "cpu"
        persist_dir: Override default persist directory.
        collection_name: Override default collection name.

    Returns:
        Chroma vectorstore ready for queries
    """
    _persist = str(persist_dir) if persist_dir is not None else PERSIST_DIR
    _collection = collection_name if collection_name is not None else COLLECTION_NAME

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
        collection_name=_collection,
        persist_directory=_persist,
    )
    return vectorstore


# Backward-compatible alias
build_index = build_chroma_index
