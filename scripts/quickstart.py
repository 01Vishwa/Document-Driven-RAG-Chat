#!/usr/bin/env python
"""
Aviation RAG Chat - Quick Start Script
Run this to set up and start the system
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))


def check_env():
    """Check if .env file exists and has required variables."""
    env_file = project_dir / ".env"
    env_example = project_dir / ".env.example"
    
    if not env_file.exists():
        print("⚠️  .env file not found!")
        if env_example.exists():
            print("   Copy .env.example to .env and configure it:")
            print("   copy .env.example .env")
        return False
    
    # Check for GITHUB_TOKEN
    with open(env_file, 'r') as f:
        content = f.read()
    
    if "GITHUB_TOKEN=" not in content or "your_github_token_here" in content:
        print("⚠️  GITHUB_TOKEN not configured in .env!")
        print("   Edit .env and add your GitHub Personal Access Token")
        return False
    
    return True


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import fastapi
        import streamlit
        import sentence_transformers
        import faiss
        print("✅ Dependencies installed")
        return True
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False


def check_documents():
    """Check if documents are available for ingestion."""
    raw_dirs = [
        project_dir / "data" / "raw",
        project_dir / "Raw",
    ]
    
    for raw_dir in raw_dirs:
        if raw_dir.exists():
            pdfs = list(raw_dir.rglob("*.pdf"))
            if pdfs:
                print(f"✅ Found {len(pdfs)} PDF documents in {raw_dir}")
                return True
    
    print("⚠️  No PDF files found in data/raw or Raw directories")
    return False


def check_index():
    """Check if FAISS index exists."""
    index_dirs = [
        project_dir / "data" / "index",
        project_dir / "data" / "faiss_index",
    ]
    
    for index_dir in index_dirs:
        if index_dir.exists() and (index_dir / "faiss.index").exists():
            print("✅ FAISS index found")
            return True
    
    print("ℹ️  No FAISS index found (run ingestion first)")
    return False


def run_ingestion():
    """Run document ingestion."""
    print("\n" + "=" * 50)
    print("Running Document Ingestion...")
    print("=" * 50)
    subprocess.run([sys.executable, str(project_dir / "scripts" / "run_ingest.py")])


def run_server():
    """Run the FastAPI server."""
    print("\n" + "=" * 50)
    print("Starting Server...")
    print("=" * 50)
    print("FastAPI:   http://localhost:8000")
    print("API Docs:  http://localhost:8000/docs")
    print("=" * 50)
    subprocess.run([sys.executable, str(project_dir / "scripts" / "run_api.py")])


def main():
    print("=" * 60)
    print("Aviation RAG Chat - Quick Start")
    print("=" * 60)
    print()
    
    # Run checks
    print("Running system checks...\n")
    
    env_ok = check_env()
    deps_ok = check_dependencies()
    docs_ok = check_documents()
    index_ok = check_index()
    
    print()
    
    if not env_ok or not deps_ok:
        print("❌ Please fix the issues above before continuing")
        return 1
    
    # If no index, offer to run ingestion
    if not index_ok and docs_ok:
        response = input("Would you like to run document ingestion? [y/N]: ")
        if response.lower() == 'y':
            run_ingestion()
    
    # Start server
    response = input("\nWould you like to start the API server? [Y/n]: ")
    if response.lower() != 'n':
        run_server()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
