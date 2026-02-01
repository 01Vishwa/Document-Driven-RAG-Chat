"""
Aviation RAG Chat - Embedding Service
GPU-accelerated embeddings using sentence-transformers
"""

import torch
import numpy as np
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from loguru import logger

from app.core.config import settings


class EmbeddingService:
    """
    Generate embeddings using sentence-transformers with GPU acceleration.
    
    Features:
    - Automatic GPU detection and utilization
    - Batch processing for efficiency
    - Caching of model instance
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding service."""
        if self._model is None:
            self._initialize_model()
    
    def _initialize_model(self):
        """Load the embedding model with GPU support."""
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.info("Running on CPU (GPU not available or disabled)")
        
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        
        try:
            self._model = SentenceTransformer(
                settings.embedding_model,
                device=self.device
            )
            
            # Warm up the model
            _ = self._model.encode(["warmup"], show_progress_bar=False)
            
            logger.info(
                f"Embedding model loaded successfully on {self.device}. "
                f"Dimension: {self._model.get_sentence_embedding_dimension()}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the model instance."""
        if self._model is None:
            self._initialize_model()
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embedding
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
        )
        return embedding
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (N x dimension)
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            device=self.device,
        )
        
        logger.info(f"Generated embeddings: shape {embeddings.shape}")
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query (may use different prefix/format).
        
        Some models benefit from different formatting for queries vs documents.
        
        Args:
            query: Query text
            
        Returns:
            Numpy array of query embedding
        """
        # For most sentence-transformers, query embedding is the same as document
        # Some models (like E5) use prefixes - add here if needed
        return self.embed_text(query)
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Since embeddings are normalized, dot product = cosine similarity.
        
        Args:
            query_embedding: Single query embedding
            document_embeddings: Matrix of document embeddings
            
        Returns:
            Array of similarity scores
        """
        # Ensure proper shapes
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Dot product with normalized vectors = cosine similarity
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        
        return similarities


# Singleton instance
embedding_service = EmbeddingService()
