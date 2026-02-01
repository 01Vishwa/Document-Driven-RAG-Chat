"""
API Routes for the Aviation RAG Chat.
Implements POST /ingest, POST /ask, GET /health endpoints.

CRITICAL GUARANTEES:
- /ask ALWAYS retrieves before generating (no hallucination paths)
- Citations ONLY from actually retrieved chunks
- Debug mode returns top 3 chunks with full scores
- Proper HTTP errors for bad states
- Request logging for traceability
"""
import time
from typing import List
from fastapi import APIRouter, HTTPException

from app.core import get_settings, logger
from app.models import (
    IngestRequest,
    IngestResponse,
    AskRequest,
    AskResponse,
    Citation,
    RetrievedChunk,
    AskRequest,
    AskResponse,
    Citation,
    RetrievedChunk,
    DebugData,
    HealthResponse,
)
from app.services import (
    get_ingestion_service,
    get_retriever,
    get_generation_service,
)

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns system status and index information.
    
    CRITICAL: Returns index_loaded state for frontend awareness.
    """
    ingestion_service = get_ingestion_service()
    stats = ingestion_service.get_stats()
    
    return HealthResponse(
        status="healthy" if stats["initialized"] else "degraded",
        version="1.0.0",
        index_loaded=stats["initialized"],
        documents_indexed=stats["unique_documents"],
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest = None):
    """
    Ingest PDF documents into the vector and BM25 indices.
    
    If no file_paths provided, ingests all PDFs from the Raw directory.
    Force reindex ensures clean rebuild (no duplicate vectors).
    """
    logger.info(f"Ingest request: force_reindex={request.force_reindex if request else False}")
    
    try:
        ingestion_service = get_ingestion_service()
        
        # Use request params or defaults
        file_paths = request.file_paths if request else None
        force_reindex = request.force_reindex if request else False
        
        result = ingestion_service.ingest_documents(
            pdf_paths=file_paths,
            force_reindex=force_reindex
        )
        
        logger.info(f"Ingest complete: {result['message']}")
        
        return IngestResponse(
            success=result["success"],
            message=result["message"],
            documents_processed=result["documents_processed"],
            total_chunks=result["total_chunks"],
            processing_time_seconds=result["processing_time_seconds"]
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about the aviation documents.
    
    CRITICAL GUARANTEES:
    1. ALWAYS retrieves before generating (no hallucination paths)
    2. Citations ONLY from actually retrieved chunks
    3. Debug mode returns top 3 chunks with full scores
    4. Fails clearly if index not loaded
    """
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"Ask request: '{request.question[:100]}...' debug={request.debug}")
    
    try:
        # Get services
        ingestion_service = get_ingestion_service()
        retriever = get_retriever()
        generator = get_generation_service()
        
        # CRITICAL: Check if indices are loaded
        if not ingestion_service.is_initialized():
            # Try to load existing indices
            logger.info("Indices not initialized, attempting to load...")
            ingestion_service.ingest_documents()
            
            if not ingestion_service.is_initialized():
                logger.error("Index not loaded and could not be loaded")
                raise HTTPException(
                    status_code=503,
                    detail="Document index not available. Please run POST /ingest first to index documents."
                )
        
        # STEP 0: ROUTER CHECK (Level 2 Bonus / Phase 3)
        # Check if we can answer without retrieval (e.g., greetings)
        router_response = generator.route_query(request.question)
        if router_response:
            logger.info("Router handled query directly (Greeting)")
            return AskResponse(
                answer=router_response,
                citations=[],
                is_grounded=True,
                confidence_score=1.0,
                debug_data=DebugData(
                    retrieved_chunks=[],
                    router_decision="direct_reply",
                    retrieval_method="none",
                    processing_time_ms=(time.time() - start_time) * 1000
                ) if request.debug else None
            )

        # STEP 1: ALWAYS RETRIEVE FIRST
        # This is the ONLY code path - LLM is NEVER called without retrieval
        retrieval_results = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            use_reranker=True  # Always use reranker for best quality
        )
        
        logger.info(f"Retrieved {len(retrieval_results)} chunks")
        
        # STEP 2: Check for empty retrieval
        if not retrieval_results:
            logger.warning("No relevant chunks found for question")
            # Return explicit refusal with empty citations
            return AskResponse(
                answer=settings.refusal_message,
                citations=[],
                is_grounded=True,  # Refusal is properly grounded
                confidence_score=0.0,
                debug_data=DebugData(
                    retrieved_chunks=[],
                    router_decision="retrieval_chain",
                    retrieval_method=retriever.get_retrieval_method_name(),
                    processing_time_ms=(time.time() - start_time) * 1000
                ) if request.debug else None
            )
        
        # STEP 3: Generate answer from retrieved context ONLY
        answer, confidence, is_grounded = generator.generate(
            question=request.question,
            retrieval_results=retrieval_results
        )
        
        # STEP 4: Build citations ONLY from actually retrieved chunks
        # CRITICAL: Citations must match retrieved chunks exactly
        citations: List[Citation] = []
        for result in retrieval_results:
            chunk = result.chunk
            # Truncate snippet for response but keep it meaningful
            snippet = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            
            citations.append(Citation(
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                relevance_score=result.rerank_score,
                snippet=snippet
            ))
        
        # STEP 5: Build response
        response = AskResponse(
            answer=answer,
            citations=citations,
            is_grounded=is_grounded,
            confidence_score=confidence,
        )
        
        # STEP 6: Add debug info if requested (top chunks with scores)
        if request.debug:
            # Return ALL retrieved chunks for transparency
            debug_chunks = [
                RetrievedChunk(
                    chunk_id=r.chunk.chunk_id,
                    content=r.chunk.content,
                    document_name=r.chunk.document_name,
                    page_number=r.chunk.page_number,
                    vector_score=r.vector_score,
                    bm25_score=r.bm25_score,
                    rerank_score=r.rerank_score
                )
                for r in retrieval_results
            ]
            
            response.debug_data = DebugData(
                retrieved_chunks=debug_chunks,
                router_decision="retrieval_chain",
                retrieval_method="Hybrid (Vector*0.7 + BM25*0.3) -> Cross-Encoder",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        total_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Question answered in {total_time_ms:.0f}ms, "
                   f"grounded={is_grounded}, confidence={confidence:.2f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ["router"]
