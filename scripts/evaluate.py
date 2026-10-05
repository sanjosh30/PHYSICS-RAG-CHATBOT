"""
scripts/evaluate.py
--------------------
CLI script to run the 20-question benchmark evaluation.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --output logs/my_eval.json
    python scripts/evaluate.py --hallucination-check
"""

import argparse
import io
import json
import sys
from pathlib import Path

# Force UTF-8 on Windows consoles (cp1252 can't encode emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import run_benchmark
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_report(metrics: dict) -> None:
    """Pretty-print the evaluation report."""
    print(f"\n{'='*60}")
    print("  Physics RAG Chatbot — Evaluation Report")
    print(f"{'='*60}\n")

    print(f"  Total questions:          {metrics['total']}")
    print(f"  Physics questions:        {metrics['physics_questions']}")
    print(f"  Out-of-scope questions:   {metrics['out_of_scope_questions']}")
    print()

    ca = metrics["citation_accuracy"]
    hr = metrics["hallucination_rate"]
    rr = metrics["refusal_rate"]

    ca_icon = "✅" if ca >= 0.85 else "❌"
    hr_icon = "✅" if hr < 0.10 else "❌"
    rr_icon = "✅" if rr >= 0.90 else "❌"

    print(f"  {ca_icon} Citation Accuracy:        {ca:.1%}  (target ≥ 85%)")
    print(f"  {hr_icon} Hallucination Rate:       {hr:.1%}  (target < 10%)")
    print(f"  {rr_icon} OOS Refusal Rate:         {rr:.1%}  (target ≥ 90%)")
    print(f"     Mean Confidence:         {metrics['mean_confidence']:.1%}")
    print()

    targets = metrics.get("targets_met", {})
    all_met = all(targets.values())
    if all_met:
        print("  🎉 ALL TARGETS MET — Production ready!")
    else:
        failed = [k for k, v in targets.items() if not v]
        print(f"  ⚠️  Targets not met: {', '.join(failed)}")

    print()
    print("  Per-category breakdown:")
    for cat, cat_metrics in metrics.get("per_category", {}).items():
        print(
            f"    {cat:20s}  "
            f"n={cat_metrics['count']}  "
            f"citation_acc={cat_metrics['citation_accuracy']:.0%}  "
            f"conf={cat_metrics['mean_confidence']:.0%}"
        )
    print(f"\n{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Physics RAG benchmark evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="logs/evaluation_results.json",
        help="Output path for evaluation results JSON.",
    )
    parser.add_argument(
        "--hallucination-check",
        action="store_true",
        help="Run LLM-as-judge hallucination detection after benchmark.",
    )
    args = parser.parse_args()

    print("\n[PHYSICS RAG]  Starting Evaluation\n")
    results, metrics = run_benchmark(output_path=args.output)
    print_report(metrics)

    if args.hallucination_check:
        print("Running LLM-as-judge hallucination check...\n")
        from src.evaluation.hallucination_test import HallucinationJudge
        judge = HallucinationJudge()
        halluc_results = judge.run_on_results_file(args.output)
        summary = halluc_results["summary"]
        status = "PASS" if summary["target_met"] else "FAIL"
        print(
            f"  [{status}] LLM-judged Hallucination Rate: "
            f"{summary['hallucination_rate_llm_judge']:.1%} "
            f"({summary['hallucinated_count']}/{summary['total_physics_questions']} flagged)"
        )

    print(f"\nFull results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
