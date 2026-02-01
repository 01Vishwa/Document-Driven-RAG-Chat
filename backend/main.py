"""
Aviation RAG Chat API - Main Entry Point.
FastAPI application for document-driven aviation Q&A with strict grounding.
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.core import get_settings, logger
from app.api import router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting Aviation RAG Chat API...")
    
    # Create data directory
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    # Try to load existing indices on startup
    from app.services import get_ingestion_service
    ingestion_service = get_ingestion_service()
    
    try:
        ingestion_service.ingest_documents()
        stats = ingestion_service.get_stats()
        logger.info(f"Loaded {stats['total_chunks']} chunks from {stats['unique_documents']} documents")
    except Exception as e:
        logger.warning(f"Could not load indices on startup: {e}")
        logger.info("Run POST /ingest to initialize the system")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Aviation RAG Chat API...")


# Create FastAPI app
app = FastAPI(
    title="Aviation Document AI Chat",
    description="""
    A Retrieval-Augmented Generation (RAG) system for aviation documents.
    
    Features:
    - **Hybrid Retrieval**: Combines BM25 keyword search with dense vector similarity
    - **Cross-Encoder Reranking**: Improves retrieval precision
    - **Strict Grounding**: Answers only from provided documents
    - **Hallucination Control**: Refuses to answer when context is insufficient
    
    Built for AIRMAN AI/ML Technical Assignment.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG Chat"])

# Also mount at root for convenience
app.include_router(router, tags=["RAG Chat (Root)"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Aviation Document AI Chat",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ingest": "POST /ingest",
            "ask": "POST /ask",
            "docs": "/docs"
        },
        "level": "Level 2 - Hybrid Retrieval (BM25 + Vector + Reranker)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
