"""
tests/test_generation.py
--------------------------
Unit tests for generation components: prompts, citation engine, RAG chain.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.documents import Document

from src.generation.citation_engine import CitationEngine, Citation
from src.generation.prompts import (
    format_context,
    is_out_of_scope_response,
    extract_refusal_reason,
)


def make_doc(content: str, book: str = "Test Book", chapter: str = "Ch 1",
             page: int = 10, corpus: str = "feynman") -> Document:
    return Document(
        page_content=content,
        metadata={
            "book": book, "chapter": chapter,
            "page": page, "corpus": corpus, "chunk_id": "test",
        },
    )


# ── Prompt tests ───────────────────────────────────────────────────────────────

class TestPrompts:
    def test_format_context_single_doc(self):
        docs = [(make_doc("F = ma is Newton's second law."), 0.92)]
        context = format_context(docs)
        assert "[Source 1]" in context
        assert "Test Book" in context
        assert "F = ma" in context

    def test_format_context_multiple_docs(self):
        docs = [
            (make_doc("First passage."), 0.9),
            (make_doc("Second passage."), 0.7),
        ]
        context = format_context(docs)
        assert "[Source 1]" in context
        assert "[Source 2]" in context

    def test_out_of_scope_detection_true(self):
        response = "OUT_OF_SCOPE: This is not about physics."
        assert is_out_of_scope_response(response) is True

    def test_out_of_scope_detection_false(self):
        response = "The speed of light is c = 3×10⁸ m/s [Source 1]."
        assert is_out_of_scope_response(response) is False

    def test_extract_refusal_reason(self):
        response = "OUT_OF_SCOPE: Question is not related to physics."
        reason = extract_refusal_reason(response)
        assert reason == "Question is not related to physics."
        assert "OUT_OF_SCOPE" not in reason


# ── Citation engine tests ──────────────────────────────────────────────────────

class TestCitationEngine:
    def setup_method(self):
        self.engine = CitationEngine()
        self.docs = [
            (make_doc("Newton's second law: F = ma.", page=15), 0.92),
            (make_doc("The photoelectric effect: E = hf.", page=87), 0.75),
            (make_doc("Maxwell's equations describe EM fields.", page=202), 0.61),
        ]

    def test_extract_single_citation(self):
        response = "Newton's second law is F = ma [Source 1]."
        citations = self.engine.extract(response, self.docs)
        assert len(citations) == 1
        assert citations[0].index == 1
        assert citations[0].page == 15

    def test_extract_multiple_citations(self):
        response = "F = ma [Source 1] and E = hf [Source 2]."
        citations = self.engine.extract(response, self.docs)
        assert len(citations) == 2
        indices = [c.index for c in citations]
        assert 1 in indices and 2 in indices

    def test_citation_out_of_range_ignored(self):
        response = "Some claim [Source 99]."
        citations = self.engine.extract(response, self.docs)
        assert len(citations) == 0  # Source 99 doesn't exist

    def test_no_citations_in_response(self):
        response = "The answer is 42."
        citations = self.engine.extract(response, self.docs)
        assert len(citations) == 0

    def test_aggregate_confidence(self):
        conf = self.engine.aggregate_confidence(self.docs)
        expected = (0.92 + 0.75 + 0.61) / 3
        assert abs(conf - expected) < 1e-6

    def test_aggregate_confidence_empty(self):
        assert self.engine.aggregate_confidence([]) == 0.0

    def test_citation_to_dict(self):
        cit = Citation(
            index=1, book="Test", chapter="Ch 1", page=5,
            corpus="feynman", snippet="test snippet", score=0.9
        )
        d = cit.to_dict()
        assert d["index"] == 1
        assert d["book"] == "Test"
        assert d["score"] == 0.9

    def test_citation_short_ref(self):
        cit = Citation(
            index=1, book="Feynman", chapter="Chapter 1", page=15,
            corpus="feynman", snippet="", score=0.8
        )
        assert "Feynman" in cit.short_ref
        assert "p.15" in cit.short_ref
