"""
Aviation RAG Chat - Data Models
Pydantic models for documents, chunks, and API responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# =============================================================================
# Document Models
# =============================================================================

class DocumentMetadata(BaseModel):
    """Metadata for a processed document."""
    filename: str
    filepath: str
    file_size: int
    total_pages: int
    created_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    document_type: str = "pdf"
    category: Optional[str] = None  # e.g., "Air Navigation", "Meteorology"


class TextChunk(BaseModel):
    """A chunk of text extracted from a document."""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    chunk_index: int  # Index within the document
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "doc1_chunk_0",
                "document_id": "doc1",
                "content": "The altimeter is an instrument...",
                "page_number": 15,
                "chunk_index": 0,
                "metadata": {"section": "Instruments"}
            }
        }


class ProcessedDocument(BaseModel):
    """A fully processed document with all its chunks."""
    document_id: str
    metadata: DocumentMetadata
    chunks: List[TextChunk]
    total_chunks: int
    processing_time_seconds: float


# =============================================================================
# Retrieval Models
# =============================================================================

class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store."""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    score: float  # Similarity score
    rerank_score: Optional[float] = None  # Score after reranking
    source_file: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Citation for an answer."""
    document_name: str
    page_number: int
    chunk_id: str
    relevant_text: str  # Short snippet of relevant text
    confidence: float


# =============================================================================
# API Request/Response Models
# =============================================================================

class IngestRequest(BaseModel):
    """Request to ingest documents."""
    document_paths: Optional[List[str]] = None  # If None, ingest all from Raw/
    force_reindex: bool = False


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    status: str
    documents_processed: int
    total_chunks: int
    processing_time_seconds: float
    errors: List[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    """Request to ask a question."""
    question: str
    debug: bool = False  # If True, include retrieved chunks in response
    top_k: int = 5  # Number of chunks to retrieve
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the purpose of an altimeter?",
                "debug": True,
                "top_k": 5
            }
        }


class AskResponse(BaseModel):
    """Response to a question."""
    answer: str
    citations: List[Citation]
    confidence: float
    is_grounded: bool  # Whether the answer is fully grounded in sources
    retrieved_chunks: Optional[List[RetrievedChunk]] = None  # Only if debug=True
    processing_time_seconds: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "An altimeter is used to measure altitude...",
                "citations": [
                    {
                        "document_name": "Instruments.pdf",
                        "page_number": 15,
                        "chunk_id": "instruments_chunk_42",
                        "relevant_text": "The altimeter measures...",
                        "confidence": 0.92
                    }
                ],
                "confidence": 0.92,
                "is_grounded": True,
                "processing_time_seconds": 1.23
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime
    components: Dict[str, str]


# =============================================================================
# Evaluation Models
# =============================================================================

class QuestionType(str, Enum):
    """Types of evaluation questions."""
    FACTUAL = "factual"
    APPLIED = "applied"
    REASONING = "reasoning"


class EvaluationQuestion(BaseModel):
    """A question for evaluation."""
    question_id: str
    question: str
    question_type: QuestionType
    expected_answer: Optional[str] = None
    source_page: Optional[int] = None
    source_document: Optional[str] = None


class EvaluationResult(BaseModel):
    """Result of evaluating a single question."""
    question_id: str
    question: str
    question_type: QuestionType
    generated_answer: str
    expected_answer: Optional[str]
    is_grounded: bool
    retrieval_hit: bool  # Did retrieved chunks contain the answer?
    is_faithful: bool  # Is answer faithful to sources?
    is_hallucination: bool
    confidence: float
    citations: List[Citation]
    notes: Optional[str] = None


class EvaluationReport(BaseModel):
    """Complete evaluation report."""
    total_questions: int
    by_type: Dict[str, int]
    retrieval_hit_rate: float
    faithfulness_rate: float
    hallucination_rate: float
    average_confidence: float
    results: List[EvaluationResult]
    best_answers: List[EvaluationResult]
    worst_answers: List[EvaluationResult]
    generated_at: datetime = Field(default_factory=datetime.now)
