"""Factory to create a shared LLM instance.

``llama-cpp-python`` is an optional dependency: when it is not installed
(e.g. on Streamlit Cloud) the factory returns ``None`` and all agents fall
back to mock mode automatically.
"""

from __future__ import annotations


def create_llm(
    model_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    verbose: bool = False,
    chat_format: str = "chatml",
) -> object | None:
    """Creates a llama-cpp Llama instance, or returns None if unavailable.

    Args:
        model_path:    Path to the ``.gguf`` model file.
        n_ctx:         Context window size (tokens).
        n_gpu_layers:  ``-1`` offloads all layers to Metal/MPS; ``0`` for CPU.
        verbose:       Enable llama.cpp verbose logging.
        chat_format:   Chat template format (e.g. ``"chatml"``).

    Returns:
        A ``Llama`` instance ready for inference, or ``None`` when
        ``llama-cpp-python`` is not installed or the model file is missing.
    """
    try:
        from llama_cpp import Llama

        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=4,
            verbose=verbose,
            chat_format=chat_format,
        )
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "llama-cpp-python is not installed — running in mock mode."
        )
        return None
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Failed to load LLM from %s: %s — running in mock mode.", model_path, exc
        )
        return None
