"""
Aviation RAG Chat - Retrieval Service
FAISS-based vector storage with GPU acceleration, BM25 hybrid search, and reranking
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger

import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.schemas import TextChunk, RetrievedChunk
from app.rag.embedding import embedding_service


@dataclass
class ChunkMetadata:
    """Stored metadata for each chunk."""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    chunk_index: int
    filename: str
    metadata: Dict[str, Any]


class VectorStore:
    """
    FAISS-based vector store with hybrid BM25 search.
    
    Features:
    - GPU-accelerated FAISS index
    - BM25 keyword search for hybrid retrieval
    - Persistent storage and loading
    - Metadata management
    """
    
    def __init__(
        self,
        index_path: str = None,
        use_gpu: bool = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            index_path: Path to store/load index
            use_gpu: Whether to use GPU for FAISS
        """
        self.index_path = Path(index_path or settings.faiss_index_path)
        if not self.index_path.is_absolute():
            self.index_path = settings.base_dir / self.index_path
        self.use_gpu = use_gpu if use_gpu is not None else settings.use_gpu
        
        self.dimension = embedding_service.dimension
        
        # Initialize components
        self.index: Optional[faiss.Index] = None
        self.gpu_index: Optional[faiss.Index] = None
        self.metadata: List[ChunkMetadata] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        
        # GPU resources
        self.gpu_resources = None
        
        # Try to load existing index
        if self._index_exists():
            self.load()
        else:
            self._create_empty_index()
    
    def _index_exists(self) -> bool:
        """Check if index files exist."""
        index_file = self.index_path / "faiss.index"
        meta_file = self.index_path / "metadata.pkl"
        return index_file.exists() and meta_file.exists()
    
    def _create_empty_index(self):
        """Create an empty FAISS index."""
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Optionally use IVF for larger datasets
        # For small datasets, flat index is faster
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.gpu_index = faiss.index_cpu_to_gpu(
                    self.gpu_resources, 0, self.index
                )
                logger.info("FAISS index created on GPU")
            except Exception as e:
                logger.warning(f"Failed to create GPU index, using CPU: {e}")
                self.gpu_index = None
        else:
            logger.info("FAISS index created on CPU")
    
    @property
    def active_index(self) -> faiss.Index:
        """Get the active index (GPU if available, else CPU)."""
        return self.gpu_index if self.gpu_index is not None else self.index
    
    def add_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 100,
        update_bm25: bool = True,
    ) -> int:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for embedding generation
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract texts for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = embedding_service.embed_texts(texts, batch_size=batch_size)
        
        # Add to FAISS index
        if self.gpu_index is not None:
            # GPU index
            self.gpu_index.add(embeddings.astype(np.float32))
            # Keep CPU index in sync for saving
            self.index.add(embeddings.astype(np.float32))
        else:
            self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        for chunk in chunks:
            meta = ChunkMetadata(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                filename=chunk.metadata.get("filename", "unknown"),
                metadata=chunk.metadata,
            )
            self.metadata.append(meta)
        
        # Update BM25 index
        if update_bm25:
            self._update_bm25(texts)
        else:
            # Just add to corpus for later building
            new_tokenized = [self._tokenize(text) for text in texts]
            self.tokenized_corpus.extend(new_tokenized)
        
        logger.info(
            f"Vector store now contains {self.active_index.ntotal} vectors"
        )
        
        return len(chunks)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _update_bm25(self, texts: List[str]):
        """Update BM25 index with new texts."""
        # Tokenize new texts
        new_tokenized = [self._tokenize(text) for text in texts]
        self.tokenized_corpus.extend(new_tokenized)
        
        # Rebuild BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
    def rebuild_bm25(self):
        """Rebuild BM25 index from full corpus."""
        if self.tokenized_corpus:
            logger.info(f"Rebuilding BM25 index with {len(self.tokenized_corpus)} documents")
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            logger.warning("No corpus to build BM25 index")
    
    def search_vector(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Search using vector similarity only.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (index, score) tuples
        """
        if self.active_index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = embedding_service.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        scores, indices = self.active_index.search(query_embedding, min(top_k, self.active_index.ntotal))
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((int(idx), float(score)))
        
        return results
    
    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Search using BM25 only.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (index, score) tuples
        """
        if self.bm25 is None or not self.tokenized_corpus:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        
        return results
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = None,
        bm25_weight: float = None,
    ) -> List[RetrievedChunk]:
        """
        Hybrid search combining vector and BM25 results.
        
        Uses Reciprocal Rank Fusion (RRF) to combine results.
        
        Args:
            query: Query text
            top_k: Number of final results
            vector_weight: Weight for vector results
            bm25_weight: Weight for BM25 results
            
        Returns:
            List of RetrievedChunk objects
        """
        vector_weight = vector_weight or settings.vector_weight
        bm25_weight = bm25_weight or settings.bm25_weight
        
        # Get results from both methods
        vector_results = self.search_vector(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2) if settings.use_bm25 else []
        
        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        rrf_scores: Dict[int, float] = {}
        
        # Add vector scores
        for rank, (idx, score) in enumerate(vector_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + vector_weight / (k + rank + 1)
        
        # Add BM25 scores
        for rank, (idx, score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_weight / (k + rank + 1)
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to RetrievedChunk objects
        results = []
        for idx, rrf_score in sorted_indices[:top_k]:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                
                # Get original vector score for this index
                vector_score = next(
                    (s for i, s in vector_results if i == idx),
                    0.0
                )
                
                results.append(RetrievedChunk(
                    chunk_id=meta.chunk_id,
                    document_id=meta.document_id,
                    content=meta.content,
                    page_number=meta.page_number,
                    score=vector_score,
                    source_file=meta.filename,
                    metadata=meta.metadata,
                ))
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = None,
        use_hybrid: bool = None,
    ) -> List[RetrievedChunk]:
        """
        Main search method.
        
        Args:
            query: Query text
            top_k: Number of results
            use_hybrid: Whether to use hybrid search
            
        Returns:
            List of RetrievedChunk objects
        """
        top_k = top_k or settings.top_k_retrieval
        use_hybrid = use_hybrid if use_hybrid is not None else settings.use_bm25
        
        if use_hybrid:
            return self.search_hybrid(query, top_k=top_k)
        
        # Vector-only search
        vector_results = self.search_vector(query, top_k=top_k)
        
        results = []
        for idx, score in vector_results:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(RetrievedChunk(
                    chunk_id=meta.chunk_id,
                    document_id=meta.document_id,
                    content=meta.content,
                    page_number=meta.page_number,
                    score=score,
                    source_file=meta.filename,
                    metadata=meta.metadata,
                ))
        
        return results
    
    def save(self):
        """Save the index and metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index (CPU version)
        index_file = self.index_path / "faiss.index"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        meta_file = self.index_path / "metadata.pkl"
        with open(meta_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save BM25 corpus
        bm25_file = self.index_path / "bm25_corpus.pkl"
        with open(bm25_file, 'wb') as f:
            pickle.dump(self.tokenized_corpus, f)
        
        logger.info(f"Vector store saved to {self.index_path}")
    
    def load(self):
        """Load the index and metadata from disk."""
        index_file = self.index_path / "faiss.index"
        meta_file = self.index_path / "metadata.pkl"
        bm25_file = self.index_path / "bm25_corpus.pkl"
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_file))
        
        # Move to GPU if available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.gpu_index = faiss.index_cpu_to_gpu(
                    self.gpu_resources, 0, self.index
                )
                logger.info("Loaded FAISS index to GPU")
            except Exception as e:
                logger.warning(f"Failed to load index to GPU: {e}")
        
        # Load metadata
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load BM25 corpus
        if bm25_file.exists():
            with open(bm25_file, 'rb') as f:
                self.tokenized_corpus = pickle.load(f)
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(
            f"Vector store loaded: {self.active_index.ntotal} vectors, "
            f"{len(self.metadata)} metadata entries"
        )
    
    def clear(self):
        """Clear all data from the vector store."""
        self._create_empty_index()
        self.metadata = []
        self.tokenized_corpus = []
        self.bm25 = None
        
        # Remove saved files
        if self.index_path.exists():
            for f in self.index_path.glob("*"):
                f.unlink()
        
        logger.info("Vector store cleared")
    
    @property
    def total_vectors(self) -> int:
        """Get total number of vectors in the store."""
        return self.active_index.ntotal if self.active_index else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_vectors": self.total_vectors,
            "total_chunks": len(self.metadata),
            "dimension": self.dimension,
            "using_gpu": self.gpu_index is not None,
            "has_bm25": self.bm25 is not None,
            "index_path": str(self.index_path),
        }


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


# Singleton instances
vector_store = VectorStore()
reranker_service = RerankerService()
