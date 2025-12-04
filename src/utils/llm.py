"""LLM utility functions for loading HuggingFace models."""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from src.config import settings

_small_llm_cache: ChatHuggingFace | None = None
_large_llm_cache: ChatHuggingFace | None = None


def _load_model(model_path: str, model_type: str) -> ChatHuggingFace:
    """Internal helper to load a HuggingFace model."""
    print(f"Loading {model_type} HuggingFace model from: {model_path}")
    
    llm_pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_path,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "do_sample": True,
            "return_full_text": False,
        },
        model_kwargs={
            "trust_remote_code": True,
            "device_map": "auto",
        }
    )
    
    llm = ChatHuggingFace(llm=llm_pipeline)
    print(f"✓ {model_type} HuggingFace model loaded successfully")
    
    return llm


def get_small_model() -> ChatHuggingFace:
    """Get or create small HuggingFace LLM singleton (for router)."""
    global _small_llm_cache
    if _small_llm_cache is None:
        _small_llm_cache = _load_model(settings.llm_model_small, "Small")
    return _small_llm_cache


def get_large_model() -> ChatHuggingFace:
    """Get or create large HuggingFace LLM singleton (for RAG and logic)."""
    global _large_llm_cache
    if _large_llm_cache is None:
        _large_llm_cache = _load_model(settings.llm_model_large, "Large")
    return _large_llm_cache

