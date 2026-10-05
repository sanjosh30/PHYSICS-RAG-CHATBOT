"""
scripts/ingest.py
------------------
CLI script to ingest physics PDFs into the ChromaDB vector store.

Usage:
    python scripts/ingest.py                    # ingest all
    python scripts/ingest.py --corpus feynman   # feynman only
    python scripts/ingest.py --corpus openstax  # openstax only
    python scripts/ingest.py --batch-size 64    # smaller batches
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.build_index import build_index
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest physics PDFs into the ChromaDB vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        choices=["feynman", "openstax", "all"],
        default="all",
        help="Which corpus to ingest.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of chunks per ChromaDB batch.",
    )
    parser.add_argument(
        "--skip-pages",
        type=int,
        default=8,
        help="Number of front-matter pages to skip per PDF.",
    )
    args = parser.parse_args()

    logger.info(f"Starting ingestion: corpus={args.corpus}")
    print(f"\n{'='*60}")
    print(f"  Physics RAG Chatbot — Document Ingestion")
    print(f"  Corpus: {args.corpus.upper()}")
    print(f"{'='*60}\n")

    store = build_index(
        corpus=args.corpus,
        batch_size=args.batch_size,
        skip_pages=args.skip_pages,
    )

    store.print_stats()
    print("\n✅ Ingestion complete! Run the app with:")
    print("   streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    main()
