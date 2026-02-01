"""
Aviation RAG Chat - RAG Engine
Main orchestrator for retrieval-augmented generation
"""

import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from app.core.config import settings
from app.models.schemas import (
    TextChunk,
    ProcessedDocument,
    RetrievedChunk,
    Citation,
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
)
from app.services.pdf_processor import pdf_processor
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store
from app.services.reranker import reranker_service
from app.services.llm import llm_service


class RAGEngine:
    """
    Main RAG engine orchestrating the full pipeline:
    1. Document ingestion
    2. Query processing
    3. Retrieval (hybrid)
    4. Reranking
    5. Grounded answer generation
    """
    
    def __init__(self):
        """Initialize the RAG engine."""
        self.pdf_processor = pdf_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker_service
        self.llm = llm_service
        
        logger.info("RAG Engine initialized")
    
    # =========================================================================
    # Ingestion
    # =========================================================================
    
    def ingest_documents(
        self,
        document_paths: Optional[List[str]] = None,
        force_reindex: bool = False,
    ) -> IngestResponse:
        """
        Ingest PDF documents into the vector store.
        
        Args:
            document_paths: List of PDF paths (if None, use default directory)
            force_reindex: Whether to clear existing index first
            
        Returns:
            IngestResponse with statistics
        """
        start_time = time.time()
        errors = []
        
        if force_reindex:
            logger.info("Force reindex: clearing existing vector store")
            self.vector_store.clear()
        
        # Determine documents to process
        if document_paths:
            pdf_paths = [Path(p) for p in document_paths if p.endswith('.pdf')]
        else:
            # Use default raw documents directory
            raw_dir = settings.raw_docs_dir
            logger.info(f"Scanning for PDFs in: {raw_dir}")
            pdf_paths = list(raw_dir.rglob("*.pdf"))
        
        if not pdf_paths:
            return IngestResponse(
                status="warning",
                documents_processed=0,
                total_chunks=0,
                processing_time_seconds=time.time() - start_time,
                errors=["No PDF files found"],
            )
        
        logger.info(f"Found {len(pdf_paths)} PDF files to process")
        
        # Process documents
        total_chunks = 0
        processed_count = 0
        
        for pdf_path in pdf_paths:
            try:
                # Process PDF
                processed_doc = self.pdf_processor.process_document(str(pdf_path))
                
                # Add chunks to vector store
                chunks_added = self.vector_store.add_chunks(processed_doc.chunks)
                total_chunks += chunks_added
                processed_count += 1
                
                logger.info(
                    f"Processed {pdf_path.name}: "
                    f"{processed_doc.total_chunks} chunks"
                )
                
            except Exception as e:
                error_msg = f"Failed to process {pdf_path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Save the vector store
        self.vector_store.save()
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Ingestion complete: {processed_count} documents, "
            f"{total_chunks} chunks, {processing_time:.2f}s"
        )
        
        return IngestResponse(
            status="success" if not errors else "partial",
            documents_processed=processed_count,
            total_chunks=total_chunks,
            processing_time_seconds=processing_time,
            errors=errors,
        )
    
    # =========================================================================
    # Query Pipeline
    # =========================================================================
    
    def ask(
        self,
        question: str,
        top_k: int = None,
        debug: bool = False,
    ) -> AskResponse:
        """
        Process a question through the full RAG pipeline.
        
        Pipeline:
        1. Embed query
        2. Hybrid retrieval (vector + BM25)
        3. Rerank results
        4. Generate grounded answer
        5. Verify grounding
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            debug: Whether to include retrieved chunks in response
            
        Returns:
            AskResponse with answer and citations
        """
        start_time = time.time()
        
        top_k = top_k or settings.top_k_retrieval
        
        # Check if we have any documents
        if self.vector_store.total_vectors == 0:
            return AskResponse(
                answer="No documents have been ingested yet. Please ingest aviation documents first.",
                citations=[],
                confidence=0.0,
                is_grounded=True,
                retrieved_chunks=None,
                processing_time_seconds=time.time() - start_time,
            )
        
        logger.info(f"Processing question: {question[:100]}...")
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(
            query=question,
            top_k=top_k,
            use_hybrid=settings.use_bm25,
        )
        
        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 2: Rerank if enabled
        if settings.use_reranker and self.reranker.is_available:
            retrieved_chunks = self.reranker.rerank(
                query=question,
                chunks=retrieved_chunks,
                top_k=settings.top_k_rerank,
            )
            logger.debug(f"Reranked to {len(retrieved_chunks)} chunks")
        
        # Step 3: Check relevance threshold
        if not retrieved_chunks or all(
            chunk.score < settings.similarity_threshold 
            for chunk in retrieved_chunks
        ):
            return AskResponse(
                answer="This information is not available in the provided document(s).",
                citations=[],
                confidence=0.0,
                is_grounded=True,
                retrieved_chunks=retrieved_chunks if debug else None,
                processing_time_seconds=time.time() - start_time,
            )
        
        # Step 4: Generate grounded answer
        generation_result = self.llm.generate_answer(
            question=question,
            chunks=retrieved_chunks,
            check_grounding=settings.grounding_check,
        )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Answer generated: grounded={generation_result.is_grounded}, "
            f"confidence={generation_result.confidence:.2f}, "
            f"time={processing_time:.2f}s"
        )
        
        return AskResponse(
            answer=generation_result.answer,
            citations=generation_result.citations,
            confidence=generation_result.confidence,
            is_grounded=generation_result.is_grounded,
            retrieved_chunks=retrieved_chunks if debug else None,
            processing_time_seconds=processing_time,
        )
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "reranker_enabled": settings.use_reranker,
            "bm25_enabled": settings.use_bm25,
            "grounding_check_enabled": settings.grounding_check,
        }
    
    def health_check(self) -> Dict[str, str]:
        """Check health of all components."""
        components = {}
        
        # Embedding service
        try:
            _ = self.embedding_service.embed_text("test")
            components["embeddings"] = "healthy"
        except Exception as e:
            components["embeddings"] = f"unhealthy: {e}"
        
        # Vector store
        try:
            _ = self.vector_store.total_vectors
            components["vector_store"] = "healthy"
        except Exception as e:
            components["vector_store"] = f"unhealthy: {e}"
        
        # LLM
        if self.llm.health_check():
            components["llm"] = "healthy"
        else:
            components["llm"] = "unhealthy"
        
        # Reranker
        if settings.use_reranker:
            if self.reranker.is_available:
                components["reranker"] = "healthy"
            else:
                components["reranker"] = "disabled or unavailable"
        else:
            components["reranker"] = "disabled"
        
        return components


# Singleton instance
rag_engine = RAGEngine()
