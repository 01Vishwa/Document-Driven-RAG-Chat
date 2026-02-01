"""
Aviation RAG Chat - Core Configuration Package
"""

from .config import settings, get_settings
from .schemas import (
    DocumentMetadata,
    TextChunk,
    ProcessedDocument,
    RetrievedChunk,
    Citation,
    IngestRequest,
    IngestResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    QuestionType,
    EvaluationQuestion,
    EvaluationResult,
    EvaluationReport,
)

__all__ = [
    "settings",
    "get_settings",
    "DocumentMetadata",
    "TextChunk",
    "ProcessedDocument",
    "RetrievedChunk",
    "Citation",
    "IngestRequest",
    "IngestResponse",
    "AskRequest",
    "AskResponse",
    "HealthResponse",
    "QuestionType",
    "EvaluationQuestion",
    "EvaluationResult",
    "EvaluationReport",
]
