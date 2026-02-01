#!/usr/bin/env python
"""
Aviation RAG Chat - Document Ingestion Script
Ingest PDF documents into the vector store
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from app.services.rag_engine import rag_engine
from app.core.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Ingest aviation PDF documents into the vector store"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to PDF file or directory (default: Raw/ directory)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing (clear existing index)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics and exit"
    )
    
    args = parser.parse_args()
    
    # Show stats only
    if args.stats:
        stats = rag_engine.get_stats()
        print("\n" + "=" * 50)
        print("VECTOR STORE STATISTICS")
        print("=" * 50)
        for key, value in stats.get("vector_store", {}).items():
            print(f"{key}: {value}")
        print("=" * 50)
        return
    
    # Determine paths
    document_paths = None
    if args.path:
        path = Path(args.path)
        if path.is_file() and path.suffix.lower() == '.pdf':
            document_paths = [str(path)]
        elif path.is_dir():
            document_paths = [str(p) for p in path.rglob("*.pdf")]
        else:
            logger.error(f"Invalid path: {args.path}")
            sys.exit(1)
    
    # Log source
    if document_paths:
        logger.info(f"Ingesting from: {args.path}")
        logger.info(f"Found {len(document_paths)} PDF files")
    else:
        logger.info(f"Ingesting from default: {settings.raw_docs_dir}")
    
    # Run ingestion
    logger.info("Starting document ingestion...")
    
    response = rag_engine.ingest_documents(
        document_paths=document_paths,
        force_reindex=args.force,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("INGESTION RESULTS")
    print("=" * 50)
    print(f"Status:              {response.status}")
    print(f"Documents Processed: {response.documents_processed}")
    print(f"Total Chunks:        {response.total_chunks}")
    print(f"Processing Time:     {response.processing_time_seconds:.2f}s")
    
    if response.errors:
        print(f"\nErrors ({len(response.errors)}):")
        for error in response.errors:
            print(f"  - {error}")
    
    print("=" * 50)
    
    # Show final stats
    stats = rag_engine.get_stats()
    print(f"\nVector Store: {stats['vector_store']['total_vectors']} vectors indexed")


if __name__ == "__main__":
    main()
