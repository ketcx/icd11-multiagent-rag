"""Downloads the LLM and embedding models declared in configs/app.yaml.

Run via:
    python main.py download_models
or directly:
    python scripts/download_models.py
"""

from __future__ import annotations

from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer


def _load_config(config_path: str = "configs/app.yaml") -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def download_llm(config: dict | None = None) -> str:
    """Downloads the GGUF LLM declared in ``configs/app.yaml``.

    Args:
        config: Pre-loaded config dict. Loaded from disk when omitted.

    Returns:
        Local path of the downloaded GGUF file.
    """
    cfg = config or _load_config()
    repo_id: str = cfg["llm"]["model_name"]
    filename: str = cfg["llm"]["model_file"]

    print(f"Downloading LLM: {repo_id} / {filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"  ✓ LLM cached at: {path}")
    return path


def download_embeddings(config: dict | None = None) -> SentenceTransformer:
    """Downloads and caches the embedding model declared in ``configs/app.yaml``.

    Args:
        config: Pre-loaded config dict. Loaded from disk when omitted.

    Returns:
        Loaded ``SentenceTransformer`` instance.
    """
    cfg = config or _load_config()
    model_name: str = cfg["embeddings"]["model_name"]

    print(f"Downloading embeddings: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"  ✓ Embeddings loaded: {dim} dimensions")
    return model


if __name__ == "__main__":
    config = _load_config()
    download_llm(config)
    download_embeddings(config)
