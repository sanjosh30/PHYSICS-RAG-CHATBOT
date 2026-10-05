"""
src/ingestion/chunker.py
------------------------
Splits cleaned page texts into overlapping chunks suitable for embedding.

Strategy: RecursiveCharacterTextSplitter that tries to split at paragraph
boundaries first, then sentence boundaries, then words.  Each chunk inherits
the metadata of its parent page plus a unique chunk_id.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_chunk_id(text: str, metadata: dict[str, Any]) -> str:
    """Deterministic chunk ID based on content + source + page."""
    key = f"{metadata.get('source', '')}::{metadata.get('page', 0)}::{text[:80]}"
    return hashlib.md5(key.encode()).hexdigest()


def chunk_documents(
    docs: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a list of page dicts into LangChain Document chunks.

    Args:
        docs:          List of page dicts {text, metadata}.
        chunk_size:    Characters per chunk (defaults to settings value).
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of LangChain Document objects with enriched metadata.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
        is_separator_regex=False,
        keep_separator=False,
        strip_whitespace=True,
    )

    all_chunks: list[Document] = []

    for doc in docs:
        text = doc.get("text", "").strip()
        metadata = doc.get("metadata", {})

        if not text:
            continue

        raw_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < 30:
                continue  # skip trivially short chunks

            chunk_meta = {
                **metadata,
                "chunk_index": i,
                "chunk_id": _make_chunk_id(chunk_text, metadata),
            }
            all_chunks.append(
                Document(page_content=chunk_text, metadata=chunk_meta)
            )

    logger.info(
        f"Chunking: {len(docs)} pages → {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks
