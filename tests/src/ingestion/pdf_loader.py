"""
src/ingestion/pdf_loader.py
---------------------------
Loads physics PDFs using PyMuPDF and extracts structured text with metadata.

Metadata attached to each page:
  - source:  absolute path to the PDF file
  - book:    human-readable book title
  - chapter: chapter heading extracted from the page (best-effort)
  - page:    1-indexed page number
  - corpus:  "feynman" | "openstax"
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Map filename keywords → canonical book title
_TITLE_MAP: dict[str, str] = {
    "vol 1": "Feynman Lectures Vol 1 — Mechanics, Radiation & Heat",
    "vol 2": "Feynman Lectures Vol 2 — Electromagnetism & Matter",
    "vol 3": "Feynman Lectures Vol 3 — Quantum Mechanics",
    "university-physics-v1": "OpenStax University Physics Vol 1",
    "university-physics-v2": "OpenStax University Physics Vol 2",
    "university-physics-v3": "OpenStax University Physics Vol 3",
}


def _infer_title(pdf_path: Path) -> str:
    """Infer a human-readable book title from the filename."""
    stem = pdf_path.stem.lower()
    for key, title in _TITLE_MAP.items():
        if key in stem:
            return title
    return pdf_path.stem  # fallback to raw stem


def _infer_corpus(pdf_path: Path) -> str:
    """Infer corpus name ('feynman' or 'openstax') from directory."""
    parts = [p.lower() for p in pdf_path.parts]
    if "feynman" in parts:
        return "feynman"
    if "openstax" in parts:
        return "openstax"
    return "unknown"


def _extract_chapter(page_text: str) -> str:
    """
    Best-effort chapter extraction.
    Looks for patterns like 'Chapter 12', '12-1', 'CHAPTER IV', etc.
    """
    # Pattern: 'Chapter N' or 'CHAPTER N'
    m = re.search(r"(?:Chapter|CHAPTER)\s+(\d+[A-Za-z]?)", page_text)
    if m:
        return f"Chapter {m.group(1)}"
    # Feynman-style: '12-1 Title of section' at start of line
    m = re.search(r"^(\d{1,2}-\d)", page_text, re.MULTILINE)
    if m:
        return f"Section {m.group(1)}"
    return "Unknown Chapter"


def load_pdf(
    pdf_path: Path,
    skip_pages: int = 0,
) -> Generator[dict, None, None]:
    """
    Yield one dict per page:
        {
            "text": str,
            "metadata": {
                "source":  str,
                "book":    str,
                "chapter": str,
                "page":    int,
                "corpus":  str,
            }
        }

    Args:
        pdf_path:   Path to the PDF file.
        skip_pages: Number of front-matter pages to skip (title, TOC, etc.).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    book_title = _infer_title(pdf_path)
    corpus = _infer_corpus(pdf_path)
    current_chapter = "Unknown Chapter"

    logger.info(f"Loading PDF: {pdf_path.name} ({corpus})")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    logger.info(f"  → {total_pages} pages found")

    for page_num in range(total_pages):
        if page_num < skip_pages:
            continue

        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction

        if not text or len(text.strip()) < 50:
            # Skip near-empty pages (images, blank pages)
            continue

        # Update chapter tracker when a chapter heading is found
        detected = _extract_chapter(text)
        if detected != "Unknown Chapter":
            current_chapter = detected

        yield {
            "text": text,
            "metadata": {
                "source": str(pdf_path),
                "book": book_title,
                "chapter": current_chapter,
                "page": page_num + 1,  # 1-indexed
                "corpus": corpus,
            },
        }

    doc.close()
    logger.info(f"  → Finished loading {pdf_path.name}")


def load_corpus_dir(
    corpus_dir: Path,
    skip_pages: int = 8,
) -> Generator[dict, None, None]:
    """
    Load all PDFs from a directory, yielding page dicts.

    Args:
        corpus_dir: Directory containing PDF files.
        skip_pages: Pages to skip per PDF (front matter).
    """
    corpus_dir = Path(corpus_dir)
    pdf_files = sorted(corpus_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in: {corpus_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) in {corpus_dir}")

    for pdf_path in pdf_files:
        yield from load_pdf(pdf_path, skip_pages=skip_pages)
