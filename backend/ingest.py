"""
Ingestion Script - Command Line Tool.
Ingests aviation PDFs from the Raw directory into the vector and BM25 indices.
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from app.core import logger
from app.services import get_ingestion_service


def main():
    """Run ingestion on all PDFs."""
    print("="*60)
    print("Aviation RAG Chat - Document Ingestion")
    print("="*60)
    
    ingestion_service = get_ingestion_service()
    
    # Check for Raw directory
    raw_dir = Path("Raw")
    if not raw_dir.exists():
        raw_dir = Path("../Raw")
    
    if raw_dir.exists():
        pdfs = list(raw_dir.rglob("*.pdf"))
        print(f"\nFound {len(pdfs)} PDF files in {raw_dir.absolute()}:")
        for pdf in pdfs:
            print(f"  - {pdf.name} ({pdf.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("\nNo Raw directory found. Place PDFs in ./Raw/")
    
    print("\nStarting ingestion...")
    result = ingestion_service.ingest_documents(force_reindex=True)
    
    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"Success:            {result['success']}")
    print(f"Documents Processed: {result['documents_processed']}")
    print(f"Total Chunks:       {result['total_chunks']}")
    print(f"Processing Time:    {result['processing_time_seconds']:.2f}s")
    print("="*60)
    
    if result['success']:
        print("\nIndices saved to ./data/")
        print("You can now run the API with: python main.py")


if __name__ == "__main__":
    main()
