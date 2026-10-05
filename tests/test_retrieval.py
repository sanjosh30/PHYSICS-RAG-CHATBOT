"""
tests/test_retrieval.py
------------------------
Unit tests for the vector store and retrieval pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.retrieval.reranker import Reranker
from src.utils.helpers import sigmoid, format_confidence, truncate_text


# ── Helper factories ───────────────────────────────────────────────────────────

def make_doc(content: str, metadata: dict | None = None) -> Document:
    return Document(
        page_content=content,
        metadata=metadata or {
            "book": "Test Book",
            "chapter": "Chapter 1",
            "page": 42,
            "corpus": "feynman",
            "chunk_id": "abc123",
        },
    )


# ── Helpers tests ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_sigmoid_zero(self):
        assert abs(sigmoid(0.0) - 0.5) < 1e-6

    def test_sigmoid_positive(self):
        assert sigmoid(5.0) > 0.99

    def test_sigmoid_negative(self):
        assert sigmoid(-5.0) < 0.01

    def test_truncate_short(self):
        assert truncate_text("hello world", 100) == "hello world"

    def test_truncate_long(self):
        text = "a " * 200
        result = truncate_text(text, 50)
        assert len(result) <= 55  # some tolerance for word boundary
        assert result.endswith("…")

    def test_format_confidence_high(self):
        assert "High" in format_confidence(0.9)

    def test_format_confidence_medium(self):
        assert "Medium" in format_confidence(0.65)

    def test_format_confidence_low(self):
        assert "Low" in format_confidence(0.2)


# ── Reranker tests ─────────────────────────────────────────────────────────────

class TestReranker:
    @patch("src.retrieval.reranker._get_cross_encoder")
    def test_rerank_orders_by_score(self, mock_get_encoder):
        """Reranker should return docs sorted by confidence descending."""
        mock_encoder = MagicMock()
        import numpy as np
        # High score for first doc, low for second
        mock_encoder.predict.return_value = np.array([2.5, -1.0])
        mock_get_encoder.return_value = mock_encoder

        reranker = Reranker()
        docs = [
            (make_doc("Newton's second law F=ma"), 0.9),
            (make_doc("Chocolate cake recipe"), 0.1),
        ]
        result = reranker.rerank("What is Newton's second law?", docs, top_n=2)

        assert len(result) == 2
        # First result should have higher confidence
        assert result[0][1] > result[1][1]

    @patch("src.retrieval.reranker._get_cross_encoder")
    def test_rerank_top_n_limit(self, mock_get_encoder):
        """Reranker should return at most top_n results."""
        mock_encoder = MagicMock()
        import numpy as np
        mock_encoder.predict.return_value = np.array([1.0, 0.5, -0.5])
        mock_get_encoder.return_value = mock_encoder

        reranker = Reranker()
        docs = [(make_doc(f"Doc {i}"), 0.5) for i in range(3)]
        result = reranker.rerank("test query", docs, top_n=2)

        assert len(result) == 2

    @patch("src.retrieval.reranker._get_cross_encoder")
    def test_is_in_scope_high_confidence(self, mock_get_encoder):
        """is_in_scope should return True when max confidence ≥ threshold."""
        mock_encoder = MagicMock()
        import numpy as np
        mock_encoder.predict.return_value = np.array([3.0])
        mock_get_encoder.return_value = mock_encoder

        reranker = Reranker()
        docs = [(make_doc("Physics content"), 0.9)]
        reranked = reranker.rerank("physics question", docs, top_n=1)

        assert reranker.is_in_scope(reranked) is True

    @patch("src.retrieval.reranker._get_cross_encoder")
    def test_is_out_of_scope_low_confidence(self, mock_get_encoder):
        """is_in_scope should return False for very low confidence."""
        mock_encoder = MagicMock()
        import numpy as np
        mock_encoder.predict.return_value = np.array([-5.0])
        mock_get_encoder.return_value = mock_encoder

        reranker = Reranker()
        docs = [(make_doc("Unrelated content"), 0.1)]
        reranked = reranker.rerank("what is bitcoin?", docs, top_n=1)

        assert reranker.is_in_scope(reranked) is False

    def test_rerank_empty_input(self):
        """Reranker should handle empty input gracefully."""
        with patch("src.retrieval.reranker._get_cross_encoder") as mock:
            mock.return_value = MagicMock()
            reranker = Reranker()
            result = reranker.rerank("query", [], top_n=3)
            assert result == []
