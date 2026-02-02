"""
Aviation RAG Chat - Configuration Management
Centralized configuration using Pydantic Settings
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==========================================================================
    # LLM Configuration
    # ==========================================================================
    github_token: str = Field(default="", env="GITHUB_TOKEN")
    llm_endpoint: str = Field(
        default="https://models.github.ai/inference",
        env="LLM_ENDPOINT"
    )
    llm_model: str = Field(default="openai/gpt-4o", env="LLM_MODEL")
    
    # ==========================================================================
    # Embedding Configuration
    # ==========================================================================
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # ==========================================================================
    # FAISS Configuration
    # ==========================================================================
    faiss_index_path: str = Field(default="./data/faiss_index", env="FAISS_INDEX_PATH")
    use_gpu: bool = Field(default=False, env="USE_GPU")
    
    # ==========================================================================
    # Chunking Configuration
    # ==========================================================================
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, env="CHUNK_OVERLAP")
    
    # ==========================================================================
    # Retrieval Configuration
    # ==========================================================================
    top_k_retrieval: int = Field(default=10, env="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, env="TOP_K_RERANK")
    similarity_threshold: float = Field(default=0.3, env="SIMILARITY_THRESHOLD")
    
    # ==========================================================================
    # Reranker Configuration
    # ==========================================================================
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        env="RERANKER_MODEL"
    )
    use_reranker: bool = Field(default=True, env="USE_RERANKER")
    
    # ==========================================================================
    # BM25 Configuration
    # ==========================================================================
    use_bm25: bool = Field(default=True, env="USE_BM25")
    bm25_weight: float = Field(default=0.3, env="BM25_WEIGHT")
    vector_weight: float = Field(default=0.7, env="VECTOR_WEIGHT")
    
    # ==========================================================================
    # Grounding Configuration
    # ==========================================================================
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    grounding_check: bool = Field(default=True, env="GROUNDING_CHECK")
    
    # ==========================================================================
    # API Configuration
    # ==========================================================================
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    
    # ==========================================================================
    # Document Paths
    # ==========================================================================
    raw_documents_path: str = Field(default="./data/raw", env="RAW_DOCUMENTS_PATH")
    processed_data_path: str = Field(default="./data/processed", env="PROCESSED_DATA_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def base_dir(self) -> Path:
        """Get the base directory of the project."""
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        data_path = self.base_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path
    
    @property
    def faiss_dir(self) -> Path:
        """Get the FAISS index directory."""
        faiss_path = Path(self.faiss_index_path)
        if not faiss_path.is_absolute():
            faiss_path = self.base_dir / faiss_path
        faiss_path.mkdir(parents=True, exist_ok=True)
        return faiss_path
    
    @property
    def raw_docs_dir(self) -> Path:
        """Get the raw documents directory."""
        raw_path = Path(self.raw_documents_path)
        if not raw_path.is_absolute():
            raw_path = self.base_dir / raw_path
        return raw_path
    
    @property
    def processed_dir(self) -> Path:
        """Get the processed data directory."""
        processed_path = Path(self.processed_data_path)
        if not processed_path.is_absolute():
            processed_path = self.base_dir / processed_path
        processed_path.mkdir(parents=True, exist_ok=True)
        return processed_path


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
