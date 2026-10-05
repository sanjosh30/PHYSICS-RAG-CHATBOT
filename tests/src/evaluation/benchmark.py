"""
src/evaluation/benchmark.py
-----------------------------
Runs the 20-question benchmark against the live RAG chain and produces
a structured evaluation report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from src.evaluation.metrics import (
    EvalResult,
    _keyword_overlap,
    compute_all_metrics,
    is_hallucination,
)
from src.generation.rag_chain import PhysicsRAGChain
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_questions(path: str | Path | None = None) -> list[dict]:
    """Load benchmark questions from JSON."""
    path = Path(path or settings.benchmark_file)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(
    questions: list[dict] | None = None,
    output_path: str | Path | None = None,
) -> tuple[list[EvalResult], dict]:
    """
    Run all benchmark questions through the RAG chain.

    Args:
        questions:   List of question dicts. If None, loads from config path.
        output_path: Where to save the results JSON.

    Returns:
        (list[EvalResult], metrics_dict)
    """
    questions = questions or load_questions()
    chain = PhysicsRAGChain()

    results: list[EvalResult] = []
    logger.bind(evaluation=True).info(
        f"Starting benchmark: {len(questions)} questions"
    )

    for q in tqdm(questions, desc="Benchmarking", unit="q"):
        qid = q["id"]
        question_text = q["question"]
        ground_truth = q["ground_truth"]
        expected_kws = q.get("expected_keywords", [])
        expected_src_kws = q.get("expected_source_keywords", [])
        category = q.get("category", "unknown")
        is_oos = category == "out_of_scope"

        logger.bind(evaluation=True).debug(f"Q{qid}: {question_text[:60]}")

        start = time.perf_counter()
        try:
            result = chain.ask(question_text)
        except Exception as e:
            logger.bind(evaluation=True).error(f"Q{qid} failed: {e}")
            results.append(
                EvalResult(
                    question_id=qid,
                    category=category,
                    question=question_text,
                    ground_truth=ground_truth,
                    model_answer=f"ERROR: {e}",
                    is_out_of_scope=True,
                    correctly_refused=is_oos,
                    citation_accuracy=0.0,
                    notes=f"Exception: {e}",
                )
            )
            continue
        elapsed = time.perf_counter() - start

        # Retrieved snippets for citation accuracy
        retrieved_snippets = [doc.page_content for doc, _ in result.retrieved_docs]

        # Citation accuracy: keyword overlap in retrieved passages
        citation_acc = _keyword_overlap(
            " ".join(retrieved_snippets), expected_src_kws
        )

        # Correctly refused?
        correctly_refused = is_oos and result.is_out_of_scope

        eval_result = EvalResult(
            question_id=qid,
            category=category,
            question=question_text,
            ground_truth=ground_truth,
            model_answer=result.answer,
            is_out_of_scope=result.is_out_of_scope,
            correctly_refused=correctly_refused,
            citation_accuracy=citation_acc if not is_oos else 1.0,
            citations=[c.to_dict() for c in result.citations],
            confidence=result.confidence,
            notes=f"Elapsed: {elapsed:.1f}s",
        )
        results.append(eval_result)

    # Compute aggregate metrics
    metrics = compute_all_metrics(results)

    # Save results
    output_path = Path(output_path or "logs/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metrics": metrics,
        "results": [
            {
                "id": r.question_id,
                "category": r.category,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "model_answer": r.model_answer,
                "is_out_of_scope": r.is_out_of_scope,
                "correctly_refused": r.correctly_refused,
                "citation_accuracy": r.citation_accuracy,
                "confidence": r.confidence,
                "num_citations": len(r.citations),
                "notes": r.notes,
            }
            for r in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.bind(evaluation=True).info(f"Results saved to {output_path}")
    return results, metrics
