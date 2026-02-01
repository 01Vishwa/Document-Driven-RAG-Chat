import os
import time
from pathlib import Path
from typing import List, Dict, Set
from prettytable import PrettyTable

# Add parent directory to path so we can import app modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services import get_ingestion_service

def generate_report(chunks: List, processing_time: float, output_file: str = "ingestion_report.md"):
    """
    Generate a detailed breakdown of the ingestion process.
    """
    stats: Dict[str, Dict] = {}
    
    # Aggregation
    for chunk in chunks:
        doc = chunk.document_name
        if doc not in stats:
            stats[doc] = {
                "pages": set(),
                "chunks": 0,
                "token_usage_est": 0
            }
        stats[doc]["chunks"] += 1
        if chunk.page_number:
            stats[doc]["pages"].add(chunk.page_number)
        stats[doc]["token_usage_est"] += len(chunk.content) / 4  # Rough est
        
    # Table for Console
    table = PrettyTable()
    table.field_names = ["Document", "Pages Processed", "Chunks Created", "Est. Tokens"]
    table.align = "l"
    
    # Markdown Content
    md_content = f"""# 📑 Ingestion Analysis Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Processing Time:** {processing_time:.2f} seconds
**Total Documents:** {len(stats)}
**Total Chunks:** {len(chunks)}

## 📊 Document Breakdown

| Document Name | Pages Detected | Chunks Created | Avg. Chunk Size (chars) |
| :--- | :---: | :---: | :---: |
"""
    
    print("\n" + "="*50)
    print("INGESTION SUMMARY")
    print("="*50)
    
    total_chunks = 0
    total_pages = 0
    
    for doc, data in sorted(stats.items()):
        page_count = len(data["pages"])
        chunk_count = data["chunks"]
        total_chunks += chunk_count
        total_pages += page_count
        
        # Calculate avg chunk size for this doc
        # (This is approximate as we didn't store total chars per doc in this simple loop, 
        # but we can infer from tokens if we want, or just skip it. 
        # Let's simple use "N/A" or iterate chunks again if needed. 
        # For efficiency, we'll just show counts.)
        
        table.add_row([doc, page_count, chunk_count, f"{data['token_usage_est']:.0f}"])
        md_content += f"| `{doc}` | {page_count} | {chunk_count} | ~{int(data['token_usage_est']*4/chunk_count) if chunk_count else 0} |\n"

    print(table)
    print(f"\nTotal: {total_pages} Pages | {total_chunks} Chunks")
    print("="*50)
    
    md_content += f"\n\n**Total:** {total_pages} Pages Processed | {total_chunks} Vectors Generated\n"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"\n📄 Detailed report saved to: {os.path.abspath(output_file)}")


def main():
    print("🚀 Starting Targeted Ingestion...")
    start_global = time.time()
    
    # target files
    # We need to resolve relative paths from the backend root
    # or absolute paths.
    # The app assumes 'Raw' is in the CWD (backend usually, or root?)
    # existing Dockerfile sets WORKDIR /app, volumes /Raw:/app/Raw
    # Code uses `settings.raw_dir` which defaults to "Raw".
    
    # Local path fix: The user is in `E:\Document-Driven-RAG-Chat`
    # The `Raw` folder is in `E:\Document-Driven-RAG-Chat\Raw`.
    # `backend` is `E:\Document-Driven-RAG-Chat\backend`.
    # So `Raw` is `../Raw`.
    
    base_raw = Path("../Raw").resolve()
    
    targets = [
        base_raw / "Air-Regulation-RK-BALI.pdf",
        base_raw / "Meteorology" / "Meteorology full book.pdf",
        base_raw / "Air Navigation" / "10-General-Navigation-2014 (1).pdf",
        base_raw / "Air Navigation" / "11-radio-navigation-2014.pdf",
        base_raw / "Air Navigation" / "6-mass-and-balance-and-performance-2014.pdf",
        base_raw / "Air Navigation" / "7-Flight-Planning-and-Monitoring-2014.pdf",
        base_raw / "Air Navigation" / "Instruments.pdf"
    ]
    
    # Convert to strings and verify existence
    valid_targets = []
    for t in targets:
        if t.exists():
            valid_targets.append(str(t))
        else:
            print(f"❌ Warning: File not found: {t}")
            
    if not valid_targets:
        print("No valid files found!")
        return
        
    print(f"found {len(valid_targets)} files for processing.")
    
    service = get_ingestion_service()
    
    # Check for existing backup to resume?
    # For now, we always start fresh as requested, but we enabled saving.
    # To resume, one would call: service.load_backup_and_upload()
    
    # Run Ingestion
    print("Triggering backend ingestion logic...")
    # NOTE: This will now SAVE a backup to data/ingestion_backup.pkl
    result = service.ingest_documents(
        pdf_paths=valid_targets,
        force_reindex=True
    )
    
    end_global = time.time()
    duration = end_global - start_global
    
    if result["success"]:
        print("✔ Ingestion Complete!")
        generate_report(service.chunks, duration)
    else:
        print(f"❌ Ingestion Failed: {result.get('message')}")

if __name__ == "__main__":
    main()
