#!/usr/bin/env python
"""
Aviation RAG Chat - Quick Test Script
Verify the system is working correctly
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json

API_BASE = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 50)
    print("Testing /health endpoint...")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        data = response.json()
        
        print(f"Status: {data.get('status', 'unknown')}")
        print("\nComponents:")
        for comp, status in data.get("components", {}).items():
            icon = "✅" if "healthy" in status else "❌"
            print(f"  {icon} {comp}: {status}")
        
        return data.get("status") == "healthy"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_stats():
    """Test stats endpoint."""
    print("\n" + "=" * 50)
    print("Testing /stats endpoint...")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE}/stats", timeout=10)
        data = response.json()
        
        print(f"Vector Store:")
        vs = data.get("vector_store", {})
        print(f"  Total vectors: {vs.get('total_vectors', 0)}")
        print(f"  Using GPU: {vs.get('using_gpu', False)}")
        print(f"  Has BM25: {vs.get('has_bm25', False)}")
        
        print(f"\nConfiguration:")
        print(f"  Embedding model: {data.get('embedding_model', 'N/A')}")
        print(f"  LLM model: {data.get('llm_model', 'N/A')}")
        print(f"  Reranker enabled: {data.get('reranker_enabled', False)}")
        print(f"  BM25 enabled: {data.get('bm25_enabled', False)}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_ask_simple():
    """Test a simple question."""
    print("\n" + "=" * 50)
    print("Testing /ask endpoint (simple question)...")
    print("=" * 50)
    
    question = "What is an altimeter?"
    print(f"Question: {question}")
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "debug": False, "top_k": 5},
            timeout=60,
        )
        data = response.json()
        
        print(f"\nAnswer: {data.get('answer', 'No answer')[:500]}...")
        print(f"\nConfidence: {data.get('confidence', 0):.1%}")
        print(f"Grounded: {data.get('is_grounded', False)}")
        print(f"Processing time: {data.get('processing_time_seconds', 0):.2f}s")
        
        if data.get("citations"):
            print(f"\nCitations:")
            for cit in data["citations"][:3]:
                print(f"  - {cit['document_name']}, Page {cit['page_number']}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_ask_refusal():
    """Test that out-of-scope questions are refused."""
    print("\n" + "=" * 50)
    print("Testing /ask endpoint (refusal behavior)...")
    print("=" * 50)
    
    question = "What is the capital of France?"
    print(f"Question: {question}")
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "debug": False, "top_k": 5},
            timeout=60,
        )
        data = response.json()
        
        answer = data.get("answer", "")
        is_refusal = "not available in the provided document" in answer.lower()
        
        print(f"\nAnswer: {answer[:300]}")
        print(f"\n{'✅ Correctly refused' if is_refusal else '❌ Should have refused'}")
        
        return is_refusal
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_documents():
    """Test documents endpoint."""
    print("\n" + "=" * 50)
    print("Testing /documents endpoint...")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE}/documents", timeout=10)
        data = response.json()
        
        print(f"Total documents: {data.get('total_documents', 0)}")
        print(f"Total chunks: {data.get('total_chunks', 0)}")
        
        if data.get("documents"):
            print("\nDocuments:")
            for doc in data["documents"][:5]:
                print(f"  - {doc['filename']} ({doc['chunk_count']} chunks)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Aviation RAG Chat - System Test")
    print("=" * 60)
    print(f"API Base URL: {API_BASE}")
    
    results = {}
    
    # Run tests
    results["health"] = test_health()
    results["stats"] = test_stats()
    results["documents"] = test_documents()
    results["ask_simple"] = test_ask_simple()
    results["ask_refusal"] = test_ask_refusal()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, passed_test in results.items():
        icon = "✅" if passed_test else "❌"
        print(f"  {icon} {test}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
