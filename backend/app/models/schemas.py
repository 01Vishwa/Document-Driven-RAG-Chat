"""
Pydantic models for API request/response schemas.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ============== Ingestion Models ==============

class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    file_paths: Optional[List[str]] = Field(
        default=None,
        description="List of file paths to ingest. If empty, ingests all PDFs from data directory."
    )
    force_reindex: bool = Field(
        default=False,
        description="Force re-indexing even if index exists."
    )


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    chunk_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    message: str
    documents_processed: int
    total_chunks: int
    processing_time_seconds: float


# ============== Chat/Ask Models ==============

class Citation(BaseModel):
    """Citation information for a response."""
    document_name: str
    page_number: Optional[int] = None
    chunk_id: str
    relevance_score: float
    snippet: str = Field(
        ...,
        description="Short text snippet from the cited chunk."
    )


class RetrievedChunk(BaseModel):
    """A retrieved chunk with metadata."""
    chunk_id: str
    content: str
    document_name: str
    page_number: Optional[int] = None
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None


class AskRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about the aviation documents."
    )
    debug: bool = Field(
        default=False,
        description="If true, returns detailed retrieval information."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top chunks to use for answer generation."
    )


class AskResponse(BaseModel):
    """Response model for a question."""
    answer: str
    citations: List[Citation]
    is_grounded: bool = Field(
        default=True,
        description="Whether the answer is fully grounded in retrieved context."
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the answer."
    )
class DebugData(BaseModel):
    """Debug information for detailed traceability."""
    retrieved_chunks: List[RetrievedChunk]
    router_decision: str
    retrieval_method: str
    processing_time_ms: float


class AskResponse(BaseModel):
    """Response model for a question."""
    answer: str
    citations: List[Citation]
    is_grounded: bool = Field(
        default=True,
        description="Whether the answer is fully grounded in retrieved context."
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the answer."
    )
    # Nested debug data (Phase 4 requirement)
    debug_data: Optional[DebugData] = Field(
        default=None,
        description="Detailed debug information (only if debug=True)."
    )


# ============== Health Check Models ==============

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = "healthy"
    version: str = "1.0.0"
    index_loaded: bool = False
    documents_indexed: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============== Evaluation Models ==============

class EvaluationQuestion(BaseModel):
    """A question for evaluation."""
    id: int
    question: str
    category: str = Field(
        ...,
        description="One of: 'factual', 'applied', 'reasoning'"
    )
    expected_answer: Optional[str] = None
    expected_grounded: bool = True


class EvaluationResult(BaseModel):
    """Result of evaluating a single question."""
    question_id: int
    question: str
    category: str
    generated_answer: str
    is_grounded: bool
    retrieval_hit: bool
    faithfulness_score: float
    citations: List[Citation]
    notes: Optional[str] = None


class EvaluationReport(BaseModel):
    """Complete evaluation report."""
    total_questions: int
    retrieval_hit_rate: float
    faithfulness_score: float
    hallucination_rate: float
    grounding_rate: float
    category_breakdown: dict
    best_answers: List[EvaluationResult]
    worst_answers: List[EvaluationResult]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
