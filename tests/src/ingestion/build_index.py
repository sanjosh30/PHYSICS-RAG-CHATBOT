"""
src/ingestion/build_index.py
-----------------------------
Orchestrates the full ingestion pipeline:
  Load PDFs → Clean → Chunk → Embed → Store in ChromaDB

Idempotent: already-indexed chunk IDs are skipped.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from src.ingestion.chunker import chunk_documents
from src.ingestion.pdf_loader import load_corpus_dir
from src.ingestion.text_cleaner import clean_documents
from src.retrieval.vector_store import PhysicsVectorStore
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_index(
    corpus: str = "all",
    batch_size: int = 128,
    skip_pages: int = 8,
) -> PhysicsVectorStore:
    """
    Run the full ingestion pipeline.

    Args:
        corpus:     Which corpus to ingest: "feynman", "openstax", or "all".
        batch_size: Number of chunks to add to ChromaDB per batch.
        skip_pages: Number of front-matter pages to skip per PDF.

    Returns:
        Initialised PhysicsVectorStore.
    """
    logger.info(f"=== Starting ingestion pipeline (corpus={corpus}) ===")

    # 1. Determine which directories to load
    dirs: list[Path] = []
    if corpus in ("feynman", "all"):
        feynman_dir = settings.feynman_dir
        if feynman_dir.exists():
            dirs.append(feynman_dir)
        else:
            logger.warning(f"Feynman directory not found: {feynman_dir}")

    if corpus in ("openstax", "all"):
        openstax_dir = settings.openstax_dir
        if openstax_dir.exists():
            dirs.append(openstax_dir)
        else:
            logger.warning(f"OpenStax directory not found: {openstax_dir}")

    if not dirs:
        raise RuntimeError("No corpus directories found. Check data/raw/ paths.")

    # 2. Load raw pages
    logger.info("Step 1/4 — Loading PDFs...")
    raw_docs: list[dict] = []
    for d in dirs:
        pages = list(load_corpus_dir(d, skip_pages=skip_pages))
        logger.info(f"  {d.name}: {len(pages)} pages loaded")
        raw_docs.extend(pages)

    logger.info(f"Total pages loaded: {len(raw_docs)}")

    # 3. Clean
    logger.info("Step 2/4 — Cleaning text...")
    cleaned_docs = clean_documents(raw_docs)

    # 4. Chunk
    logger.info("Step 3/4 — Chunking documents...")
    chunks = chunk_documents(cleaned_docs)
    logger.info(f"Total chunks created: {len(chunks)}")

    # 5. Embed and store
    logger.info("Step 4/4 — Embedding and storing in ChromaDB...")
    store = PhysicsVectorStore()

    # Get existing chunk IDs to avoid duplicates
    existing_ids = store.get_all_ids()
    logger.info(f"Existing chunks in store: {len(existing_ids)}")

    new_chunks = [
        c for c in chunks
        if c.metadata.get("chunk_id") not in existing_ids
    ]
    logger.info(f"New chunks to add: {len(new_chunks)}")

    if not new_chunks:
        logger.info("No new chunks to index. Vector store is up to date.")
        return store

    # Batch add
    for i in tqdm(
        range(0, len(new_chunks), batch_size),
        desc="Indexing batches",
        unit="batch",
    ):
        batch = new_chunks[i : i + batch_size]
        store.add_documents(batch)

    logger.info(f"=== Ingestion complete. {len(new_chunks)} chunks indexed. ===")
    store.print_stats()
    return store
