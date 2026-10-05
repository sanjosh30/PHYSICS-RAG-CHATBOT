"""
src/generation/citation_engine.py
-----------------------------------
Parses [Source N] citations from the LLM's response and maps them back
to the original retrieved document metadata to produce structured citations.

Output per citation:
  {
    "index":   int,         # [Source N] number
    "book":    str,
    "chapter": str,
    "page":    int | str,
    "corpus":  str,
    "snippet": str,         # first 200 chars of the chunk
    "score":   float,       # reranker confidence
  }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from src.utils.helpers import truncate_text
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Matches [Source N] or [Source N, M] patterns
_CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


@dataclass
class Citation:
    index: int
    book: str
    chapter: str
    page: int | str
    corpus: str
    snippet: str
    score: float
    source_path: str = field(default="")

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "book": self.book,
            "chapter": self.chapter,
            "page": self.page,
            "corpus": self.corpus,
            "snippet": self.snippet,
            "score": self.score,
            "source_path": self.source_path,
        }

    @property
    def label(self) -> str:
        return f"[Source {self.index}]"

    @property
    def short_ref(self) -> str:
        return f"{self.book}, {self.chapter}, p.{self.page}"


class CitationEngine:
    """
    Extracts and enriches citations from LLM responses.

    Usage:
        engine = CitationEngine()
        citations = engine.extract(llm_response, reranked_docs)
    """

    def extract(
        self,
        response: str,
        docs_with_scores: list[tuple[Document, float]],
    ) -> list[Citation]:
        """
        Parse [Source N] tags from the response and map to document metadata.

        Args:
            response:         The raw LLM response string.
            docs_with_scores: The reranked (Document, score) list passed to the LLM.

        Returns:
            List of Citation objects for every [Source N] tag used.
        """
        # Find all source indices referenced in the response
        mentioned = sorted(set(int(m) for m in _CITATION_PATTERN.findall(response)))

        citations: list[Citation] = []
        for idx in mentioned:
            # idx is 1-indexed
            doc_idx = idx - 1
            if doc_idx < 0 or doc_idx >= len(docs_with_scores):
                logger.warning(f"Citation [Source {idx}] out of range — skipped")
                continue

            doc, score = docs_with_scores[doc_idx]
            meta = doc.metadata

            citations.append(
                Citation(
                    index=idx,
                    book=meta.get("book", "Unknown"),
                    chapter=meta.get("chapter", "Unknown"),
                    page=meta.get("page", "?"),
                    corpus=meta.get("corpus", "unknown"),
                    snippet=truncate_text(doc.page_content, 600),
                    score=score,
                    source_path=meta.get("source", ""),
                )
            )

        logger.debug(f"Extracted {len(citations)} citation(s) from response")
        return citations

    def aggregate_confidence(
        self,
        docs_with_scores: list[tuple[Document, float]],
    ) -> float:
        """
        Compute overall answer confidence as the mean of the top doc scores.
        """
        if not docs_with_scores:
            return 0.0
        scores = [s for _, s in docs_with_scores]
        return sum(scores) / len(scores)
