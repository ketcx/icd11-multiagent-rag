"""Factory to create a shared LLM instance."""

from llama_cpp import Llama

def create_llm(model_path: str, n_ctx: int = 4096,
               n_gpu_layers: int = -1, verbose: bool = False, chat_format: str = "chatml") -> Llama:
    """Creates a llama-cpp instance with Metal support.

    Args:
        model_path: Path to the .gguf model file
        n_ctx: Context size (e.g., 4096 for OpenBioLLM-8B)
        n_gpu_layers: -1 to offload all layers to GPU (Metal)
        verbose: Verbose logging flag for llama.cpp

    Returns:
        A Llama instance ready for inference
    """
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,  # Full offload to Metal/MPS
        n_threads=4,                # Apple Silicon optimal count
        verbose=verbose,
        chat_format=chat_format,      # Use the parameter
    )
