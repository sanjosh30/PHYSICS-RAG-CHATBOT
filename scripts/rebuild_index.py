"""
scripts/rebuild_index.py
-------------------------
Wipes the ChromaDB collection and rebuilds from scratch.

Usage:
    python scripts/rebuild_index.py           # rebuild all
    python scripts/rebuild_index.py --yes     # skip confirmation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.build_index import build_index
from src.retrieval.vector_store import PhysicsVectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wipe and rebuild the ChromaDB vector index from scratch.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "--corpus",
        choices=["feynman", "openstax", "all"],
        default="all",
    )
    args = parser.parse_args()

    if not args.yes:
        print("⚠️  This will DELETE the existing vector store and rebuild from scratch.")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    print("\n🗑️  Deleting existing collection...")
    store = PhysicsVectorStore()
    store.delete_collection()
    print("✓ Collection deleted.\n")

    print("🔨 Rebuilding index...")
    build_index(corpus=args.corpus)
    print("\n✅ Index rebuilt successfully!\n")


if __name__ == "__main__":
    main()
