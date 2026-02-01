"""
Core configuration module for the Aviation RAG Chat backend.
Loads environment variables and provides settings across the application.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # GitHub Models API
    github_token: str = ""
    llm_endpoint: str = "https://models.github.ai/inference"
    llm_model: str = "openai/gpt-4o"
    
    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Reranker Model
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Data paths (from environment)
    data_dir: str = "./data"
    raw_documents_path: str = "../Raw"  # Path to PDF documents
    
    # Supabase / PGVector Settings
    supabase_connection_string: str = ""  # postgresql://postgres:password@host:port/postgres
    supabase_collection_name: str = "aviation_docs"
    
    # Deprecated (keeping for reference or cleanup)
    faiss_index_path: str = "./data/faiss_index"
    bm25_index_path: str = "./data/bm25_index.pkl"
    
    # RAG Settings
    # chunk_size=1000: Balances context completeness vs retrieval precision
    # chunk_overlap=200: Ensures sentence boundaries are preserved
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Grounding - CRITICAL for anti-hallucination
    refusal_message: str = "This information is not available in the provided document(s)."
    confidence_threshold: float = -5.0  # Rerank scores typically range -10 to 10
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def data_path(self) -> Path:
        """Return the data directory as a Path object."""
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def faiss_path(self) -> Path:
        """Return the FAISS index path."""
        return Path(self.faiss_index_path)
    
    @property
    def bm25_path(self) -> Path:
        """Return the BM25 index path."""
        return Path(self.bm25_index_path)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
