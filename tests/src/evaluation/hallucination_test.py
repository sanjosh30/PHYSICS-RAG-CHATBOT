"""
src/evaluation/hallucination_test.py
--------------------------------------
LLM-as-judge hallucination detection.

For each physics question answer, we ask the judge LLM:
  "Given only this context, does the answer contain any claims NOT supported by the context?"

Uses the same Ollama backend so no additional API cost.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a physics fact-checking judge. Given a CONTEXT (retrieved textbook passages) \
and an ANSWER from a physics tutor, determine if the ANSWER makes major factual errors \
or invents physics facts not present in the CONTEXT.

Rules:
- Minor paraphrasing or logical deduction from the context is SUPPORTED.
- Standard physics pedagogy that connects context facts is SUPPORTED.
- Only flag HALLUCINATED if the answer states a specific formula, constant, \
or factual claim that directly contradicts or is entirely absent from the CONTEXT.

Respond with ONLY one of:
  SUPPORTED   — claims are grounded in the context
  HALLUCINATED — answer contains a major invented or contradicted fact

Do NOT explain. Respond with exactly one word."""),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])


class HallucinationJudge:
    """Uses Ollama as a judge to detect hallucinations."""

    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_predict=10,
        )
        self._chain = _JUDGE_PROMPT | self._llm | StrOutputParser()

    def judge(self, context: str, answer: str) -> bool:
        """
        Returns True if the answer is hallucinated (not supported by context).
        """
        try:
            verdict = self._chain.invoke({"context": context, "answer": answer})
            verdict = verdict.strip().upper()
            return "HALLUCINATED" in verdict
        except Exception as e:
            logger.warning(f"Judge LLM failed: {e} — defaulting to NOT hallucinated")
            return False

    def run_on_results_file(
        self,
        results_path: str | Path = "logs/evaluation_results.json",
    ) -> dict:
        """
        Load existing benchmark results and run hallucination judge on each answer.
        Returns updated metrics including LLM-judged hallucination rate.
        """
        results_path = Path(results_path)
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        physics_results = [r for r in results if r["category"] != "out_of_scope"]

        hallucinated_count = 0
        judged = []

        for r in physics_results:
            if r.get("is_out_of_scope"):
                continue

            # Build context from stored citations (snippets)
            context_parts = []
            for cit in r.get("citations", []):
                # Use longer snippet (first 600 chars) for fairer judging
                snippet = cit.get("snippet", "")
                if snippet:
                    context_parts.append(snippet[:600])

            context = "\n\n".join(context_parts) or "No context available."
            answer = r.get("model_answer", "")

            is_hallucinated = self.judge(context, answer)
            if is_hallucinated:
                hallucinated_count += 1

            judged.append({**r, "llm_judged_hallucination": is_hallucinated})

        total_physics = len(physics_results)
        hallucination_rate = hallucinated_count / total_physics if total_physics else 0.0

        summary = {
            "total_physics_questions": total_physics,
            "hallucinated_count": hallucinated_count,
            "hallucination_rate_llm_judge": hallucination_rate,
            "target_met": hallucination_rate < 0.10,
        }

        logger.bind(evaluation=True).info(
            f"LLM-judged hallucination rate: {hallucination_rate:.1%} "
            f"({'✓ PASS' if summary['target_met'] else '✗ FAIL'})"
        )

        return {"summary": summary, "judged_results": judged}
