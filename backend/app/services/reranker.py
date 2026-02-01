"""
Aviation RAG Chat - Reranker Service
Cross-encoder based reranking for improved retrieval precision
"""

import torch
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from loguru import logger

from app.core.config import settings
from app.models.schemas import RetrievedChunk


class RerankerService:
    """
    Cross-encoder reranker for improving retrieval precision.
    
    Cross-encoders jointly encode query and document pairs,
    providing more accurate relevance scores than bi-encoders.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the reranker."""
        if self._model is None and settings.use_reranker:
            self._initialize_model()
    
    def _initialize_model(self):
        """Load the cross-encoder model."""
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        
        logger.info(f"Loading reranker model: {settings.reranker_model}")
        
        try:
            self._model = CrossEncoder(
                settings.reranker_model,
                max_length=512,
                device=self.device,
            )
            
            # Warm up
            _ = self._model.predict([("warmup query", "warmup document")])
            
            logger.info(f"Reranker loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            self._model = None
    
    @property
    def model(self) -> CrossEncoder:
        """Get the model instance."""
        return self._model
    
    @property
    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self._model is not None and settings.use_reranker
    
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = None,
    ) -> List[RetrievedChunk]:
        """
        Rerank retrieved chunks using cross-encoder.
        
        Args:
            query: Query text
            chunks: List of retrieved chunks
            top_k: Number of results to return
            
        Returns:
            Reranked list of chunks
        """
        if not self.is_available or not chunks:
            return chunks[:top_k] if top_k else chunks
        
        top_k = top_k or settings.top_k_rerank
        
        logger.debug(f"Reranking {len(chunks)} chunks")
        
        # Create query-document pairs
        pairs = [(query, chunk.content) for chunk in chunks]
        
        # Get reranker scores
        scores = self._model.predict(pairs)
        
        # Add rerank scores to chunks and sort
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Update chunks with rerank scores
        result = []
        for chunk, score in scored_chunks[:top_k]:
            chunk.rerank_score = float(score)
            result.append(chunk)
        
        return result
    
    def compute_relevance(self, query: str, text: str) -> float:
        """
        Compute relevance score between query and text.
        
        Args:
            query: Query text
            text: Document text
            
        Returns:
            Relevance score
        """
        if not self.is_available:
            return 0.0
        
        score = self._model.predict([(query, text)])[0]
        return float(score)


# Singleton instance
reranker_service = RerankerService()
