"""
Aviation RAG Chat - FastAPI Application
Main API endpoints for document ingestion and question answering
"""

from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
)
from app.services.rag_engine import rag_engine


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    logger.info("Aviation RAG Chat starting up...")
    logger.info(f"Vector store status: {rag_engine.vector_store.total_vectors} vectors")
    
    yield
    
    # Shutdown
    logger.info("Aviation RAG Chat shutting down...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Aviation Document AI Chat",
    description="""
    A Retrieval-Augmented Generation (RAG) system for aviation documents.
    
    Features:
    - PDF document ingestion with page-aware chunking
    - Hybrid retrieval (vector + BM25)
    - Cross-encoder reranking
    - Grounded answers with hallucination control
    - Citations with document name and page numbers
    
    **Important**: This system only answers questions from the ingested aviation documents.
    If information is not available, it will explicitly refuse to answer.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check the health status of all system components.
    
    Returns status of:
    - Embedding service
    - Vector store
    - LLM service
    - Reranker (if enabled)
    """
    components = rag_engine.health_check()
    
    overall_status = "healthy" if all(
        "healthy" in status for status in components.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.now(),
        components=components,
    )


@app.get("/stats", tags=["System"])
async def get_stats():
    """
    Get system statistics.
    
    Returns information about:
    - Number of indexed vectors/chunks
    - Model configurations
    - Feature flags
    """
    return rag_engine.get_stats()


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(request: IngestRequest = None):
    """
    Ingest PDF documents into the vector store.
    
    If no document paths are provided, processes all PDFs in the Raw/ directory.
    
    **Process:**
    1. Load PDF files
    2. Extract text with page tracking
    3. Chunk text with overlap
    4. Generate embeddings (GPU-accelerated)
    5. Store in FAISS index with BM25 index
    
    **Parameters:**
    - `document_paths`: Optional list of PDF paths to ingest
    - `force_reindex`: If true, clears existing index before ingesting
    """
    if request is None:
        request = IngestRequest()
    
    try:
        response = rag_engine.ingest_documents(
            document_paths=request.document_paths,
            force_reindex=request.force_reindex,
        )
        return response
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse, tags=["Chat"])
async def ask_question(request: AskRequest):
    """
    Ask a question and get a grounded answer from aviation documents.
    
    **Pipeline:**
    1. Query embedding generation
    2. Hybrid retrieval (vector + BM25)
    3. Cross-encoder reranking
    4. LLM answer generation with strict grounding
    5. Grounding verification
    
    **Hallucination Control:**
    If the answer cannot be supported by the retrieved documents,
    the system responds with:
    "This information is not available in the provided document(s)."
    
    **Parameters:**
    - `question`: The question to answer
    - `debug`: If true, includes retrieved chunks in response
    - `top_k`: Number of chunks to retrieve (default: 5)
    
    **Response includes:**
    - `answer`: The generated answer
    - `citations`: Document name, page number, and relevant text
    - `confidence`: Confidence score (0-1)
    - `is_grounded`: Whether the answer is fully grounded
    - `retrieved_chunks`: (debug only) The chunks used for context
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        response = rag_engine.ask(
            question=request.question,
            top_k=request.top_k,
            debug=request.debug,
        )
        return response
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """
    List all indexed documents and their statistics.
    """
    stats = rag_engine.vector_store.get_stats()
    
    # Get unique documents from metadata
    documents = {}
    for meta in rag_engine.vector_store.metadata:
        if meta.filename not in documents:
            documents[meta.filename] = {
                "filename": meta.filename,
                "document_id": meta.document_id,
                "chunk_count": 0,
            }
        documents[meta.filename]["chunk_count"] += 1
    
    return {
        "total_documents": len(documents),
        "total_chunks": stats["total_vectors"],
        "documents": list(documents.values()),
    }


@app.delete("/documents", tags=["Documents"])
async def clear_documents():
    """
    Clear all indexed documents.
    
    **Warning:** This deletes all indexed data and requires re-ingestion.
    """
    try:
        rag_engine.vector_store.clear()
        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug_mode,
    )
