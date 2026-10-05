"""
src/retrieval/reranker.py
--------------------------
Cross-encoder reranker that re-scores retrieved documents against the query.

Cross-encoders are more accurate than bi-encoders for ranking because they
process query and document together, capturing fine-grained interactions.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - Fast (6 layers) and effective for general English text
  - Returns a raw logit (unbounded float); higher = more relevant

The reranker takes top-K retrieved docs → returns top-N re-ranked with
normalised confidence scores in [0, 1].
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.utils.config import settings
from src.utils.helpers import sigmoid
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    logger.info(f"Loading cross-encoder: {settings.reranker_model}")
    model = CrossEncoder(
        settings.reranker_model,
        max_length=512,
        device=settings.embedding_device,
    )
    logger.info("Cross-encoder loaded")
    return model


class Reranker:
    """
    Re-ranks retrieved documents using a cross-encoder.

    Usage:
        reranker = Reranker()
        top_docs = reranker.rerank(query, retrieved_docs, top_n=3)
    """

    def __init__(self) -> None:
        self._model = _get_cross_encoder()

    def rerank(
        self,
        query: str,
        docs: list[tuple[Document, float]],
        top_n: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Re-rank documents and return top-n with normalised confidence scores.

        Args:
            query:  The user's question.
            docs:   List of (Document, initial_score) from the retriever.
            top_n:  Number of docs to return after reranking.

        Returns:
            List of (Document, confidence_score) sorted by confidence desc.
            confidence_score ∈ [0, 1].
        """
        top_n = top_n or settings.rerank_top_k

        if not docs:
            return []

        logger.bind(retrieval=True).debug(
            f"Reranking {len(docs)} docs → top {top_n}"
        )

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc, _ in docs]
        raw_scores: list[float] = self._model.predict(pairs).tolist()

        # Normalise with sigmoid to [0, 1]
        confidence_scores = [sigmoid(s) for s in raw_scores]

        # Sort by confidence descending
        ranked = sorted(
            zip(docs, confidence_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_docs = [(doc, conf) for (doc, _), conf in ranked[:top_n]]

        logger.bind(retrieval=True).info(
            f"Reranker top-{top_n} confidence scores: "
            + ", ".join(f"{c:.2f}" for _, c in top_docs)
        )

        return top_docs

    def max_confidence(self, reranked: list[tuple[Document, float]]) -> float:
        """Return the maximum confidence score from a reranked list."""
        if not reranked:
            return 0.0
        return max(c for _, c in reranked)

    def is_in_scope(self, reranked: list[tuple[Document, float]]) -> bool:
        """
        Return True if the query appears to be within the physics knowledge base.
        Uses the out-of-scope threshold from config.
        """
        return self.max_confidence(reranked) >= settings.out_of_scope_threshold
