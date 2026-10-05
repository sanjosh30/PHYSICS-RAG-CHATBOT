"""
src/evaluation/metrics.py
--------------------------
Evaluation metrics for the Physics RAG chatbot.

Metrics:
  1. Citation Accuracy  — Does the retrieved passage contain the expected keywords?
  2. Hallucination Rate — Does the LLM answer contradict/invent facts not in context?
  3. Out-of-Scope Refusal Rate — Does the system correctly refuse OOS questions?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    question_id: int
    category: str
    question: str
    ground_truth: str
    model_answer: str
    is_out_of_scope: bool
    correctly_refused: bool
    citation_accuracy: float       # 0.0–1.0 keyword overlap
    citations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


def _keyword_overlap(text: str, keywords: list[str]) -> float:
    """
    Compute fraction of keywords found in text (case-insensitive).
    """
    if not keywords:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def compute_citation_accuracy(
    model_answer: str,
    retrieved_snippets: list[str],
    expected_keywords: list[str],
) -> float:
    """
    Citation accuracy: check if expected keywords appear in the retrieved passages.
    This verifies the retriever found the right source, not just that the LLM said the right thing.
    """
    if not retrieved_snippets or not expected_keywords:
        return 0.0

    combined_retrieval = " ".join(retrieved_snippets)
    return _keyword_overlap(combined_retrieval, expected_keywords)


def is_hallucination(
    model_answer: str,
    retrieved_snippets: list[str],
    expected_keywords: list[str],
    threshold: float = 0.3,
) -> bool:
    """
    Simple hallucination detection: if the model answer mentions expected
    physics facts but NONE of those facts appear in the retrieved context,
    we flag it as a potential hallucination.

    This is a heuristic — a full LLM-as-judge check is in hallucination_test.py.
    """
    if not expected_keywords:
        return False

    answer_lower = model_answer.lower()
    # Check if model makes factual claims (mentions keywords)
    answer_mentions = [kw for kw in expected_keywords if kw.lower() in answer_lower]

    if not answer_mentions:
        return False  # model didn't claim anything, no hallucination

    # Check if those claims appear in context
    combined = " ".join(retrieved_snippets).lower()
    context_supports = sum(1 for kw in answer_mentions if kw.lower() in combined)

    support_ratio = context_supports / len(answer_mentions) if answer_mentions else 1.0
    return support_ratio < threshold


def compute_all_metrics(results: list[EvalResult]) -> dict:
    """
    Aggregate all metrics across the evaluation set.

    Returns:
        {
            "total": int,
            "physics_questions": int,
            "out_of_scope_questions": int,
            "citation_accuracy": float,      # % correct passage retrieval
            "hallucination_rate": float,     # % answers with suspected hallucinations
            "refusal_rate": float,           # % of OOS correctly refused
            "mean_confidence": float,
            "per_category": dict,
        }
    """
    total = len(results)
    physics_qs = [r for r in results if r.category != "out_of_scope"]
    oos_qs = [r for r in results if r.category == "out_of_scope"]

    # Citation accuracy (physics questions only)
    if physics_qs:
        citation_acc = sum(r.citation_accuracy for r in physics_qs) / len(physics_qs)
    else:
        citation_acc = 0.0

    # Refusal rate (out-of-scope questions)
    if oos_qs:
        refusal_rate = sum(1 for r in oos_qs if r.correctly_refused) / len(oos_qs)
    else:
        refusal_rate = 1.0

    # Mean confidence
    mean_conf = sum(r.confidence for r in results) / total if total > 0 else 0.0

    # Hallucination rate (simple heuristic; physics Qs only)
    # We count questions where citation_accuracy is very low but model still answered
    suspected_hallucinations = sum(
        1 for r in physics_qs
        if not r.is_out_of_scope and r.citation_accuracy < 0.25
    )
    hallucination_rate = (
        suspected_hallucinations / len(physics_qs) if physics_qs else 0.0
    )

    # Per-category breakdown
    categories = set(r.category for r in results)
    per_category: dict = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        per_category[cat] = {
            "count": len(cat_results),
            "citation_accuracy": (
                sum(r.citation_accuracy for r in cat_results) / len(cat_results)
                if cat_results else 0.0
            ),
            "mean_confidence": (
                sum(r.confidence for r in cat_results) / len(cat_results)
                if cat_results else 0.0
            ),
        }

    return {
        "total": total,
        "physics_questions": len(physics_qs),
        "out_of_scope_questions": len(oos_qs),
        "citation_accuracy": citation_acc,
        "hallucination_rate": hallucination_rate,
        "refusal_rate": refusal_rate,
        "mean_confidence": mean_conf,
        "per_category": per_category,
        "targets_met": {
            "citation_accuracy_85pct": citation_acc >= 0.85,
            "hallucination_below_10pct": hallucination_rate < 0.10,
            "refusal_90pct": refusal_rate >= 0.90,
        },
    }
