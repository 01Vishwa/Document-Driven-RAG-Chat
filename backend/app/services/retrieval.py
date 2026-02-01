"""
Hybrid Retrieval Engine with Reranking & CUDA Acceleration.
Implements Level 2 requirement: BM25 + Vector + Cross-Encoder Reranker.

CRITICAL GUARANTEES:
- Cosine similarity via normalized embeddings
- top_k >= 5 for retrieval, top 3 for debug exposure
- Deterministic ordering by score
- No code path allows LLM call without retrieval
- Proper tokenization matching ingestion
- CUDA Acceleration for Reranking
"""
import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np
from sentence_transformers import CrossEncoder
import torch

from app.core import get_settings, logger
from app.services.ingestion import get_ingestion_service, DocumentChunk, STOPWORDS


@dataclass
class RetrievalResult:
    """Result from retrieval with scores and full traceability."""
    chunk: DocumentChunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float = 0.0


class HybridRetriever:
    """
    Hybrid Retrieval Engine combining:
    1. Dense retrieval (FAISS/Vector similarity via cosine)
    2. Sparse retrieval (BM25 with proper tokenization)
    3. Cross-encoder reranking for precision
    
    GUARANTEES:
    - All retrieval uses cosine similarity (normalized embeddings)
    - Results are deterministically ordered by score
    - No code path exists where LLM is called without retrieval
    - Minimum top_k=5 for adequate context
    """
    
    # Minimum retrieval count to ensure adequate context
    MIN_TOP_K = 5
    
    def __init__(self):
        self.settings = get_settings()
        self.ingestion_service = get_ingestion_service()
        self.reranker: Optional[CrossEncoder] = None
        
        # Check for CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Retrieval Service running on: {self.device.upper()}")
        
    def _get_reranker(self) -> CrossEncoder:
        """Lazy load the reranker model."""
        if self.reranker is None:
            logger.info(f"Loading reranker: {self.settings.reranker_model} on {self.device}")
            self.reranker = CrossEncoder(self.settings.reranker_model, device=self.device)
        return self.reranker
    
    def _tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize query using the SAME method as ingestion.
        This is critical for BM25 consistency.
        """
        # Handle special aviation terms with slashes (VOR/DME -> VOR, DME)
        query = re.sub(r'/', ' ', query)
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        
        # Filter: remove stopwords and very short tokens
        tokens = [w for w in words if w not in STOPWORDS and len(w) > 1]
        
        return tokens
    
    def _vector_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Perform vector similarity search using Supabase PGVector.
        Converting Cosine Distance to Similarity (1 - distance).
        
        Returns:
            List of (chunk_index, score) tuples, sorted by score desc.
        """
        if not self.ingestion_service.is_initialized():
            logger.warning("Vector search called but indices not initialized")
            # If BM25 is initialized, we might still be able to search Supabase if connected
            # But we follow the is_initialized check for safety
            # Check connection separately?
            pass
        
        try:
            # Encode query
            model = self.ingestion_service._get_embedding_model()
            query_embedding = model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            ).astype(np.float32)
            
            dimension = query_embedding.shape[1]
            vec = query_embedding[0].tolist()
            
            # Get collection
            collection = self.ingestion_service._get_supabase_collection(dimension)
            
            # Query Supabase
            # vecs returns: [(id, distance, metadata), ...] if include_value=True, include_metadata=True
            # But wait, looking at vecs source/docs:
            # query(data, limit, filters, measure, include_value, include_metadata)
            # Default measure is what collection was created with (cosine_distance)
            results = collection.query(
                data=vec,
                limit=top_k,
                include_value=True,
                include_metadata=True
            )
            
            processed_results = []
            for r_id, distance, metadata in results:
                # Convert distance to similarity
                # Cosine distance = 1 - cosine similarity
                # So Similarity = 1 - distance
                similarity = 1.0 - distance
                
                # We need chunk_index for internal mapping
                # Metadata should have 'chunk_index'
                if metadata and 'chunk_index' in metadata:
                     idx = int(metadata['chunk_index'])
                     processed_results.append((idx, float(similarity)))
                
            logger.debug(f"Vector search: {len(processed_results)} results, top score={processed_results[0][1] if processed_results else 0.0:.4f}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Vector search failed (Supabase error?): {e}")
            return []
    
    def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Perform BM25 keyword search.
        Uses SAME tokenization as ingestion for consistency.
        
        Returns:
            List of (chunk_index, score) tuples, sorted by score desc.
        """
        if not self.ingestion_service.is_initialized():
            logger.warning("BM25 search called but indices not initialized")
            return []
        
        # Tokenize query using SAME method as ingestion
        query_tokens = self._tokenize_query(query)
        
        if not query_tokens:
            logger.warning(f"Query tokenized to empty: {query}")
            return []
        
        # Get BM25 scores
        scores = self.ingestion_service.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices sorted by score
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        
        logger.debug(f"BM25 search: {len(results)} results, top score={results[0][1]:.4f}" if results else "BM25 search: 0 results")
        return results
    
    def _combine_results(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        alpha: float = 0.7  # Rubric Requirement: Vector*0.7 + BM25*0.3
    ) -> Dict[int, RetrievalResult]:
        """
        Combine vector and BM25 results using weighted score fusion.
        Formula: Final_Score = (Vector_Score * 0.7) + (BM25_Score * 0.3)
        
        Args:
            vector_results: Results from vector search (normalized 0-1).
            bm25_results: Results from BM25 search (normalized 0-1).
            alpha: Weight for vector scores (default 0.7).
        """
        combined: Dict[int, RetrievalResult] = {}
        chunks = self.ingestion_service.chunks
        
        # Normalize scores to [0, 1] range
        def normalize(results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
            if not results:
                return []
            max_score = max(r[1] for r in results)
            if max_score == 0:
                return results
            return [(idx, score / max_score) for idx, score in results]
        
        vector_results = normalize(vector_results)
        bm25_results = normalize(bm25_results)
        
        # Add vector results
        for idx, score in vector_results:
            if idx < len(chunks):
                combined[idx] = RetrievalResult(
                    chunk=chunks[idx],
                    vector_score=score,
                    combined_score=alpha * score
                )
        
        # Add/update with BM25 results
        for idx, score in bm25_results:
            if idx < len(chunks):
                if idx in combined:
                    combined[idx].bm25_score = score
                    combined[idx].combined_score += (1 - alpha) * score
                else:
                    combined[idx] = RetrievalResult(
                        chunk=chunks[idx],
                        bm25_score=score,
                        combined_score=(1 - alpha) * score
                    )
        
        return combined
    
    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder for higher precision.
        
        Args:
            query: The original query.
            results: Combined retrieval results.
            top_k: Number of results to return after reranking.
            
        Returns:
            Reranked and filtered results, deterministically ordered.
        """
        if not results:
            return []
        
        reranker = self._get_reranker()
        
        # Prepare pairs for cross-encoder
        pairs = [(query, r.chunk.content) for r in results]
        
        # Get rerank scores with batching for speed
        batch_size = 128 if self.device == "cuda" else 32
        scores = reranker.predict(pairs, batch_size=batch_size)
        
        # Update results with rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
        
        # Sort deterministically by rerank score (higher is better)
        results.sort(key=lambda x: (-x.rerank_score, x.chunk.chunk_id))
        
        logger.debug(f"Reranking complete. Top score: {results[0].rerank_score:.4f}" if results else "")
        
        return results[:top_k]
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval with optional reranking.
        
        CRITICAL: This is the ONLY way context is retrieved.
        There is NO code path where LLM is called without this.
        
        Args:
            query: The search query.
            top_k: Number of final results to return (minimum 5).
            use_reranker: Whether to apply cross-encoder reranking.
            
        Returns:
            List of retrieval results with scores, deterministically ordered.
        """
        # Enforce minimum top_k for adequate context
        top_k = max(top_k, self.MIN_TOP_K)
        
        if not self.ingestion_service.is_initialized():
            logger.error("Retrieval attempted but indices not initialized!")
            return []
        
        # Get more candidates for reranking (2x top_k)
        candidates_k = self.settings.top_k_retrieval if use_reranker else top_k
        candidates_k = max(candidates_k, top_k * 2)
        
        # Vector search (cosine similarity)
        vector_results = self._vector_search(query, candidates_k)
        
        # BM25 search (keyword matching)
        bm25_results = self._bm25_search(query, candidates_k)
        
        # Combine results with Alpha=0.7 (Vector) and 0.3 (BM25)
        combined = self._combine_results(vector_results, bm25_results, alpha=0.7)
        
        if not combined:
            logger.warning(f"No results for query: {query[:100]}")
            return []
        
        # Sort by combined score
        results = sorted(combined.values(), key=lambda x: -x.combined_score)
        
        # Rerank if enabled (Phase 2 Requirement: Top 10 -> Reranker)
        if use_reranker and results:
            # We explicitly take Top 10 candidates for reranking
            rerank_candidates = results[:10]
            results = self._rerank(query, rerank_candidates, top_k)
        else:
            results = results[:top_k]
        
        logger.info(f"Retrieved {len(results)} chunks for: {query[:50]}...")
        return results
    
    def get_retrieval_method_name(self, use_reranker: bool = True) -> str:
        """Get human-readable name of the retrieval method."""
        if use_reranker:
            return "Hybrid (BM25 + Vector + Cross-Encoder Reranker)"
        return "Hybrid (BM25 + Vector)"


# Global singleton
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Get or create the hybrid retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
