"""
src/ingestion/text_cleaner.py
------------------------------
Cleans raw text extracted from physics PDFs.

Removes:
  - Page numbers (standalone integers on a line)
  - Running headers / footers
  - Figure and table captions
  - Excessive whitespace and hyphenation artifacts
  - Non-physics noise (copyright, ISBN, URL boilerplate)

Preserves:
  - Mathematical notation (LaTeX-like expressions, Greek letters)
  - In-line citations and equation labels
"""

from __future__ import annotations

import re

from src.utils.helpers import normalise_text
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Patterns to strip ─────────────────────────────────────────────────────────

_STANDALONE_PAGE_NUM = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

_FIGURE_CAPTION = re.compile(
    r"(?:Figure|Fig\.|Table|FIGURE|TABLE)\s+\d+[\.\d]*[^\n]*\n?",
    re.IGNORECASE,
)

_RUNNING_HEADER = re.compile(
    r"^(?:THE FEYNMAN LECTURES|UNIVERSITY PHYSICS|OpenStax)[^\n]*\n",
    re.IGNORECASE | re.MULTILINE,
)

_COPYRIGHT = re.compile(
    r"(?:©|Copyright|All rights reserved|ISBN|openstax\.org|feynmanlectures\.caltech)[^\n]*\n?",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://\S+", re.IGNORECASE)

_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")

_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")


def clean_page_text(text: str) -> str:
    """
    Apply all cleaning rules to a single page's raw text.

    Returns cleaned text, or empty string if the page is noise-only.
    """
    # Fix hyphenated line-breaks ("momen-\ntum" → "momentum")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # Remove running headers
    text = _RUNNING_HEADER.sub("", text)

    # Remove copyright/boilerplate
    text = _COPYRIGHT.sub("", text)

    # Remove URLs
    text = _URL.sub("", text)

    # Remove figure/table captions
    text = _FIGURE_CAPTION.sub("", text)

    # Remove standalone page numbers
    text = _STANDALONE_PAGE_NUM.sub("", text)

    # Collapse excessive newlines
    text = _EXCESSIVE_NEWLINES.sub("\n\n", text)

    # Unicode normalisation + whitespace cleanup
    text = normalise_text(text)

    return text


def is_useful(text: str, min_chars: int = 100) -> bool:
    """
    Return True if the cleaned text is worth indexing.
    Rejects pages that are too short after cleaning.
    """
    stripped = text.strip()
    return len(stripped) >= min_chars


def clean_document(doc: dict) -> dict | None:
    """
    Clean a single page document dict (as produced by pdf_loader).

    Returns the cleaned dict, or None if the page should be discarded.
    """
    raw_text = doc.get("text", "")
    cleaned = clean_page_text(raw_text)

    if not is_useful(cleaned):
        return None

    return {**doc, "text": cleaned}


def clean_documents(docs: list[dict]) -> list[dict]:
    """
    Clean a list of page documents, discarding empty/noise pages.
    """
    before = len(docs)
    cleaned = [clean_document(d) for d in docs]
    result = [d for d in cleaned if d is not None]
    after = len(result)
    logger.info(f"Text cleaning: {before} pages → {after} kept ({before - after} discarded)")
    return result
