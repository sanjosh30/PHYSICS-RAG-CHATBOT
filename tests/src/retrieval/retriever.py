"""
src/retrieval/retriever.py
--------------------------
High-level retriever that wraps ChromaDB with MMR (Maximal Marginal Relevance)
to return diverse, relevant chunks for a given query.

MMR balances relevance (similarity to query) with diversity (dissimilarity to
already-selected docs), reducing redundant passages in the context window.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from src.retrieval.vector_store import PhysicsVectorStore
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhysicsRetriever:
    """
    Retriever with MMR diversity and optional metadata filtering.

    Usage:
        retriever = PhysicsRetriever()
        docs = retriever.retrieve("What is Newton's second law?")
    """

    def __init__(self, store: PhysicsVectorStore | None = None) -> None:
        self._store = store or PhysicsVectorStore()
        self._chroma = self._store.as_langchain_store()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        corpus_filter: str | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve top-k relevant and diverse documents for a query.

        Args:
            query:         The user's question.
            k:             Number of documents to return.
            corpus_filter: Optionally restrict to "feynman" or "openstax".

        Returns:
            List of (Document, relevance_score) tuples, sorted by relevance desc.
        """
        k = k or settings.retrieval_top_k
        logger.bind(retrieval=True).debug(
            f"Retrieving k={k} docs for query: '{query[:80]}...'"
        )

        # Build metadata filter
        where_filter: dict[str, Any] | None = None
        if corpus_filter and corpus_filter in ("feynman", "openstax"):
            where_filter = {"corpus": corpus_filter}

        # MMR retrieval (fetch 4×k candidates, then diversify)
        fetch_k = min(k * 4, 100)

        try:
            mmr_retriever = self._chroma.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": 0.6,   # 0=max diversity, 1=max relevance
                    **({"filter": where_filter} if where_filter else {}),
                },
            )
            docs = mmr_retriever.invoke(query)

            # Pair with similarity scores (MMR doesn't return scores directly,
            # so we do a quick similarity pass to get scores for reranker input)
            scored = self._store.similarity_search(
                query, k=k, filter=where_filter
            )
            # Build score lookup by content
            score_map = {d.page_content[:100]: s for d, s in scored}

            results: list[tuple[Document, float]] = []
            for doc in docs:
                score = score_map.get(doc.page_content[:100], 0.5)
                results.append((doc, score))

        except Exception as e:
            logger.warning(f"MMR retrieval failed ({e}), falling back to similarity search")
            results = self._store.similarity_search(
                query, k=k, filter=where_filter
            )

        logger.bind(retrieval=True).info(
            f"Retrieved {len(results)} docs for query: '{query[:60]}'"
        )
        return results
