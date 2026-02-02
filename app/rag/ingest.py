"""
Aviation RAG Chat - RAG Engine
Main orchestrator for retrieval-augmented generation
"""

import os
import time
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger
import fitz  # PyMuPDF

from app.core.config import settings
from app.core.schemas import (
    TextChunk,
    ProcessedDocument,
    DocumentMetadata,
    RetrievedChunk,
    Citation,
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
)
from app.rag.ocr_extract import pdf_extractor
from app.rag.chunking import text_chunker
from app.rag.embedding import embedding_service
from app.rag.retrieval import vector_store, reranker_service
from app.rag.llm import llm_service
from app.rag.router import query_router, QueryComplexity, ConfidenceLevel


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
        self.pdf_extractor = pdf_extractor
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker_service
        self.llm = llm_service
        self.router = query_router
        
        logger.info("RAG Engine initialized with Query Router")
    
    # =========================================================================
    # Document Processing
    # =========================================================================
    
    def _generate_document_id(self, filepath: str) -> str:
        """Generate a unique document ID from filepath."""
        return hashlib.md5(filepath.encode()).hexdigest()[:12]
    
    def _infer_category(self, filepath: str) -> Optional[str]:
        """Infer document category from filepath."""
        path_lower = filepath.lower()
        
        if "navigation" in path_lower:
            return "Air Navigation"
        elif "meteorology" in path_lower:
            return "Meteorology"
        elif "regulation" in path_lower:
            return "Air Regulations"
        elif "instrument" in path_lower:
            return "Instruments"
        elif "flight-planning" in path_lower or "flight planning" in path_lower:
            return "Flight Planning"
        elif "mass" in path_lower or "balance" in path_lower:
            return "Mass and Balance"
        elif "radio" in path_lower:
            return "Radio Navigation"
        
        return None
    
    def process_document(self, pdf_path: str) -> ProcessedDocument:
        """
        Process a single PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with all chunks
        """
        start_time = time.time()
        pdf_path = str(pdf_path)
        filename = os.path.basename(pdf_path)
        
        logger.info(f"Processing document: {filename}")
        
        # Generate document ID
        doc_id = self._generate_document_id(pdf_path)
        
        # Extract pages
        pages = self.pdf_extractor.extract_all_pages(pdf_path)
        
        if not pages:
            raise ValueError(f"No text content extracted from {pdf_path}")
        
        # Get document metadata
        file_stat = os.stat(pdf_path)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        metadata = DocumentMetadata(
            filename=filename,
            filepath=pdf_path,
            file_size=file_stat.st_size,
            total_pages=total_pages,
            category=self._infer_category(pdf_path),
        )
        
        # Chunk the document
        chunks = self.text_chunker.chunk_pages(pages, doc_id, filename)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Processed {filename}: {len(chunks)} chunks, "
            f"{processing_time:.2f}s"
        )
        
        return ProcessedDocument(
            document_id=doc_id,
            metadata=metadata,
            chunks=chunks,
            total_chunks=len(chunks),
            processing_time_seconds=processing_time,
        )
    
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
                processed_doc = self.process_document(str(pdf_path))
                
                # Add chunks to vector store (defer BM25 update)
                chunks_added = self.vector_store.add_chunks(
                    processed_doc.chunks,
                    update_bm25=False
                )
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
        
        # Rebuild BM25 index once at the end
        if processed_count > 0:
            logger.info("Finalizing BM25 index...")
            self.vector_store.rebuild_bm25()
        
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
        Process a question through the full RAG pipeline with intelligent routing.
        
        Pipeline:
        1. Route query based on complexity
        2. Embed query
        3. Hybrid retrieval (vector + BM25) with routed top_k
        4. Conditional reranking based on routing decision
        5. Generate grounded answer
        6. Evaluate confidence and threshold response
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (overrides routing if provided)
            debug: Whether to include retrieved chunks in response
            
        Returns:
            AskResponse with answer and citations
        """
        start_time = time.time()
        
        # Step 0: Route the query based on complexity
        routing_decision = self.router.classify_query(question)
        logger.info(
            f"Query routed: complexity={routing_decision.complexity.value}, "
            f"reasoning={routing_decision.reasoning}"
        )
        
        # Use routed top_k unless explicitly overridden
        effective_top_k = top_k or routing_decision.retrieval_top_k
        
        # Check if we have any documents
        if self.vector_store.total_vectors == 0:
            return AskResponse(
                answer="No documents have been ingested yet. Please ingest aviation documents first.",
                citations=[],
                confidence=0.0,
                is_grounded=True,
                retrieved_chunks=None,
                processing_time_seconds=time.time() - start_time,
                routing_info={
                    "complexity": routing_decision.complexity.value,
                    "reasoning": routing_decision.reasoning,
                } if debug else None,
            )
        
        logger.info(f"Processing question: {question[:100]}...")
        
        # Step 1: Retrieve relevant chunks (top_k based on routing)
        retrieved_chunks = self.vector_store.search(
            query=question,
            top_k=effective_top_k,
            use_hybrid=settings.use_bm25,
        )
        
        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 2: Rerank based on routing decision
        use_reranker = routing_decision.should_use_reranker and self.reranker.is_available
        if use_reranker:
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
                routing_info={
                    "complexity": routing_decision.complexity.value,
                    "reasoning": routing_decision.reasoning,
                } if debug else None,
            )
        
        # Step 4: Generate grounded answer
        generation_result = self.llm.generate_answer(
            question=question,
            chunks=retrieved_chunks,
            check_grounding=settings.grounding_check,
        )
        
        # Step 5: Evaluate confidence and apply thresholding
        retrieval_scores = [chunk.score for chunk in retrieved_chunks]
        confidence_result = self.router.evaluate_confidence(
            question=question,
            answer=generation_result.answer,
            model_confidence=generation_result.confidence,
            retrieval_scores=retrieval_scores,
            is_grounded=generation_result.is_grounded,
        )
        
        logger.info(
            f"Confidence evaluation: level={confidence_result.level.value}, "
            f"score={confidence_result.score:.2f}, action={confidence_result.action}"
        )
        
        # Apply confidence-based response formatting
        final_answer = self.router.format_response_with_confidence(
            answer=generation_result.answer,
            confidence_result=confidence_result,
        )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Answer generated: grounded={generation_result.is_grounded}, "
            f"confidence={confidence_result.score:.2f}, "
            f"complexity={routing_decision.complexity.value}, "
            f"time={processing_time:.2f}s"
        )
        
        return AskResponse(
            answer=final_answer,
            citations=generation_result.citations,
            confidence=confidence_result.score,
            is_grounded=generation_result.is_grounded,
            retrieved_chunks=retrieved_chunks if debug else None,
            processing_time_seconds=processing_time,
            routing_info={
                "complexity": routing_decision.complexity.value,
                "reasoning": routing_decision.reasoning,
                "confidence_level": confidence_result.level.value,
                "confidence_action": confidence_result.action,
                "used_reranker": use_reranker,
                "effective_top_k": effective_top_k,
            } if debug else None,
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
            "query_router_enabled": True,
            "routing_features": {
                "complexity_levels": ["simple", "moderate", "complex"],
                "confidence_thresholding": True,
                "clarification_support": True,
            },
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
        
        # Query Router
        try:
            test_decision = self.router.classify_query("Test question")
            components["query_router"] = "healthy"
        except Exception as e:
            components["query_router"] = f"unhealthy: {e}"
        
        return components


# Singleton instance
rag_engine = RAGEngine()
