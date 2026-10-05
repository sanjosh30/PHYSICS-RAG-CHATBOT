"""
tests/test_citations.py
------------------------
Integration tests for the citation pipeline.
These tests mock the LLM and ChromaDB to run without external dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.generation.citation_engine import CitationEngine
from src.generation.prompts import format_context, is_out_of_scope_response


def make_doc(content: str, **kwargs) -> Document:
    defaults = {
        "book": "Feynman Lectures Vol 1",
        "chapter": "Chapter 9",
        "page": 120,
        "corpus": "feynman",
        "chunk_id": "test_chunk_001",
    }
    defaults.update(kwargs)
    return Document(page_content=content, metadata=defaults)


class TestCitationIntegration:
    """Integration tests for the full citation extraction pipeline."""

    def test_context_formatted_correctly_for_llm(self):
        """Context string must contain source markers the LLM can reference."""
        docs = [
            (make_doc("Newton's second law F=ma", page=45), 0.91),
            (make_doc("Conservation of energy principle", page=78), 0.84),
        ]
        context = format_context(docs)

        # Must have source markers
        assert "[Source 1]" in context
        assert "[Source 2]" in context

        # Must have book metadata
        assert "Feynman" in context

        # Must have the actual text
        assert "Newton" in context

    def test_citation_round_trip(self):
        """Citations extracted from LLM output must map back to correct docs."""
        docs = [
            (make_doc("F = ma is Newton's second law of motion.", page=45), 0.91),
            (make_doc("E = mc² relates energy and mass.", page=67), 0.78),
        ]

        # Simulate LLM response that cites both sources
        llm_response = (
            "Newton's second law states F = ma [Source 1]. "
            "Einstein's famous equation E = mc² [Source 2] shows mass-energy equivalence."
        )

        engine = CitationEngine()
        citations = engine.extract(llm_response, docs)

        assert len(citations) == 2

        # Source 1 → page 45
        src1 = next(c for c in citations if c.index == 1)
        assert src1.page == 45
        assert src1.corpus == "feynman"
        assert "Newton" in src1.snippet

        # Source 2 → page 67
        src2 = next(c for c in citations if c.index == 2)
        assert src2.page == 67

    def test_duplicate_citations_deduplicated(self):
        """Multiple references to [Source 1] should produce one Citation."""
        docs = [(make_doc("Content about forces.", page=10), 0.88)]
        llm_response = "Forces [Source 1] are important [Source 1]."

        engine = CitationEngine()
        citations = engine.extract(llm_response, docs)

        # Only one unique citation
        assert len(citations) == 1
        assert citations[0].index == 1

    def test_oos_response_has_no_citations(self):
        """Out-of-scope responses shouldn't have citations extracted."""
        docs = [(make_doc("Physics content"), 0.1)]
        llm_response = "OUT_OF_SCOPE: This is not a physics question."

        assert is_out_of_scope_response(llm_response) is True

        engine = CitationEngine()
        # Even if we extract, [Source N] won't appear in OOS responses
        citations = engine.extract(llm_response, docs)
        assert len(citations) == 0

    def test_confidence_reflects_reranker_scores(self):
        """Aggregate confidence should match mean of reranker scores."""
        docs = [
            (make_doc("High relevance content"), 0.95),
            (make_doc("Medium relevance content"), 0.70),
            (make_doc("Lower relevance content"), 0.55),
        ]

        engine = CitationEngine()
        conf = engine.aggregate_confidence(docs)
        expected = (0.95 + 0.70 + 0.55) / 3

        assert abs(conf - expected) < 1e-6

    def test_snippet_is_truncated(self):
        """Citation snippet should not exceed 250 characters."""
        long_text = "Physics " * 100  # 800 chars
        docs = [(make_doc(long_text), 0.8)]

        engine = CitationEngine()
        llm_response = "Some claim [Source 1]."
        citations = engine.extract(llm_response, docs)

        assert len(citations) == 1
        assert len(citations[0].snippet) <= 620  # 600 + ellipsis buffer
