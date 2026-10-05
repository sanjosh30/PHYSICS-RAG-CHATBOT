"""
src/retrieval/vector_store.py
------------------------------
ChromaDB persistent vector store wrapper.

Collection stores physics text chunks with metadata:
  source, book, chapter, page, corpus, chunk_id, chunk_index

Public API:
  store = PhysicsVectorStore()
  store.add_documents(langchain_docs)
  results = store.similarity_search(query, k=10, filter={"corpus": "feynman"})
  ids = store.get_all_ids()
  store.print_stats()
  store.delete_collection()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings.embedding_model import get_embedding_model
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhysicsVectorStore:
    """Manages the ChromaDB vector store for physics documents."""

    def __init__(self) -> None:
        self._persist_dir = settings.chroma_path
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = settings.chroma_collection_name

        self._embeddings = get_embedding_model()

        # ChromaDB 1.x PersistentClient
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))

        # LangChain Chroma wrapper (handles embedding + storage together)
        self._store = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
        )

        logger.info(
            f"ChromaDB initialised at {self._persist_dir} "
            f"(collection: {self._collection_name})"
        )

    # ── Write operations ──────────────────────────────────────────────────────

    def add_documents(self, docs: list[Document]) -> None:
        """Add LangChain Documents to the vector store."""
        if not docs:
            return

        # Use chunk_id as the stable document ID to avoid duplicates
        ids = [d.metadata.get("chunk_id", None) for d in docs]

        self._store.add_documents(documents=docs, ids=ids)
        logger.debug(f"Added {len(docs)} documents to ChromaDB")

    def delete_collection(self) -> None:
        """Wipe the entire collection (use rebuild_index to re-create)."""
        self._client.delete_collection(self._collection_name)
        logger.warning(f"Deleted collection: {self._collection_name}")

    # ── Read operations ───────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Return top-k (document, score) pairs for a query.
        Score is cosine similarity (0-1, higher = more relevant).

        Args:
            query:  Natural language query string.
            k:      Number of results (defaults to settings.retrieval_top_k).
            filter: ChromaDB where-clause metadata filter.
                    Example: {"corpus": "feynman"} or {"book": "..."}
        """
        k = k or settings.retrieval_top_k

        kwargs: dict[str, Any] = {"k": k}
        if filter:
            kwargs["filter"] = filter

        results = self._store.similarity_search_with_relevance_scores(
            query, **kwargs
        )
        return results  # list of (Document, float)

    def get_all_ids(self) -> set[str]:
        """Return the set of all chunk_ids currently in the collection."""
        try:
            collection = self._client.get_collection(self._collection_name)
            data = collection.get(include=[])
            return set(data.get("ids", []))
        except Exception:
            return set()

    def as_langchain_store(self) -> Chroma:
        """Expose the underlying LangChain Chroma object (for retriever creation)."""
        return self._store

    def print_stats(self) -> None:
        """Log collection statistics."""
        try:
            collection = self._client.get_collection(self._collection_name)
            count = collection.count()
            logger.info(
                f"ChromaDB stats - collection: {self._collection_name}, "
                f"documents: {count}"
            )
            print(f"\n  ChromaDB: {count:,} chunks in '{self._collection_name}'")
        except Exception as e:
            logger.warning(f"Could not fetch ChromaDB stats: {e}")
