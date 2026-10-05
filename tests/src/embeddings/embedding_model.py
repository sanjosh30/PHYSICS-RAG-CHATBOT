"""
src/embeddings/embedding_model.py
-----------------------------------
Wraps HuggingFace BGE-small-en-v1.5 as a LangChain Embeddings interface.

BGE (BAAI General Embedding) is optimised for retrieval tasks and produces
384-dimensional dense vectors.  It runs entirely locally — no API key needed.

Key design choices:
  - Uses HuggingFaceBgeEmbeddings which prepends the correct query instruction
    ("Represent this sentence for searching relevant passages:") automatically.
  - Batched inference with configurable batch size for memory efficiency.
  - Caches model in HuggingFace's default cache directory.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return the singleton embedding model.
    Cached so the model is only loaded once per process.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model}")

    model_kwargs = {"device": settings.embedding_device}
    encode_kwargs = {
        "normalize_embeddings": True,  # cosine similarity → dot product
        "batch_size": 32,
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    logger.info(f"Embedding model loaded on device: {settings.embedding_device}")
    return embeddings
