#!/usr/bin/env python
"""
Aviation RAG Chat - Unit Tests
Test individual components of the RAG system
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPDFExtractor:
    """Test PDF extraction functionality."""
    
    def test_clean_text(self):
        """Test text cleaning function."""
        from app.rag.ocr_extract import PDFExtractor
        
        extractor = PDFExtractor()
        
        # Test multiple spaces
        text = "This   has   multiple   spaces"
        cleaned = extractor._clean_text(text)
        assert "   " not in cleaned
        
        # Test hyphenated line breaks
        text = "word-\nbreak"
        cleaned = extractor._clean_text(text)
        assert "wordbreak" in cleaned
        
        # Test multiple newlines
        text = "Line1\n\n\n\n\nLine2"
        cleaned = extractor._clean_text(text)
        assert "\n\n\n" not in cleaned


class TestTextChunker:
    """Test text chunking functionality."""
    
    def test_generate_document_id(self):
        """Test document ID generation."""
        from app.rag.chunking import TextChunker
        
        chunker = TextChunker()
        
        doc_id1 = chunker._generate_document_id("/path/to/doc.pdf")
        doc_id2 = chunker._generate_document_id("/path/to/doc.pdf")
        doc_id3 = chunker._generate_document_id("/path/to/other.pdf")
        
        # Same path should give same ID
        assert doc_id1 == doc_id2
        # Different path should give different ID
        assert doc_id1 != doc_id3
        # ID should be 12 characters
        assert len(doc_id1) == 12


class TestVectorStore:
    """Test vector store functionality."""
    
    def test_tokenize(self):
        """Test BM25 tokenization."""
        from app.rag.retrieval import VectorStore
        
        store = VectorStore.__new__(VectorStore)
        
        text = "The aircraft's altimeter shows altitude."
        tokens = store._tokenize(text)
        
        assert "aircraft" in tokens
        assert "altimeter" in tokens
        assert "altitude" in tokens
        # Lowercase
        assert all(t.islower() for t in tokens)


class TestEmbeddings:
    """Test embedding service."""
    
    @pytest.fixture(scope="class")
    def embedding_service(self):
        """Initialize embedding service once for all tests."""
        from app.rag.embedding import EmbeddingService
        return EmbeddingService()
    
    def test_embed_text(self, embedding_service):
        """Test single text embedding."""
        embedding = embedding_service.embed_text("Test aviation document")
        
        assert embedding is not None
        assert len(embedding) == embedding_service.dimension
    
    def test_embed_texts(self, embedding_service):
        """Test batch text embedding."""
        texts = ["Text one", "Text two", "Text three"]
        embeddings = embedding_service.embed_texts(texts)
        
        assert embeddings is not None
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == embedding_service.dimension
    
    def test_embed_query(self, embedding_service):
        """Test query embedding."""
        query_embedding = embedding_service.embed_query("What is a VOR?")
        
        assert query_embedding is not None
        assert len(query_embedding) == embedding_service.dimension


class TestSchemas:
    """Test Pydantic schemas."""
    
    def test_text_chunk(self):
        """Test TextChunk schema."""
        from app.core.schemas import TextChunk
        
        chunk = TextChunk(
            chunk_id="test_chunk_001",
            document_id="test_doc",
            content="Test content for chunk",
            page_number=5,
            chunk_index=0,
        )
        
        assert chunk.chunk_id == "test_chunk_001"
        assert chunk.page_number == 5
        assert chunk.metadata == {}
    
    def test_ask_request(self):
        """Test AskRequest schema."""
        from app.core.schemas import AskRequest
        
        request = AskRequest(question="What is an altimeter?")
        
        assert request.question == "What is an altimeter?"
        assert request.debug is False
        assert request.top_k == 5
    
    def test_citation(self):
        """Test Citation schema."""
        from app.core.schemas import Citation
        
        citation = Citation(
            document_name="Instruments.pdf",
            page_number=42,
            chunk_id="inst_chunk_042",
            relevant_text="The altimeter is...",
            confidence=0.85,
        )
        
        assert citation.document_name == "Instruments.pdf"
        assert citation.confidence == 0.85


class TestConfig:
    """Test configuration loading."""
    
    def test_settings_defaults(self):
        """Test default settings values."""
        from app.core.config import settings
        
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 128
        assert settings.top_k_retrieval == 10
        assert settings.similarity_threshold == 0.3
    
    def test_paths(self):
        """Test path resolution."""
        from app.core.config import settings
        
        assert settings.base_dir.exists()
        # Data dir should be created
        assert settings.data_dir.exists()


class TestEvaluator:
    """Test evaluation system."""
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        from app.eval.evaluate import Evaluator
        
        evaluator = Evaluator()
        
        text = "The altimeter measures altitude using atmospheric pressure"
        keywords = evaluator._extract_keywords(text)
        
        assert "altimeter" in keywords
        assert "altitude" in keywords
        assert "atmospheric" in keywords
        assert "pressure" in keywords
        # Stop words should be excluded
        assert "the" not in keywords
        assert "using" not in keywords


class TestQueryRouter:
    """Test query routing functionality."""
    
    def test_simple_query_classification(self):
        """Test classification of simple queries."""
        from app.rag.router import QueryRouter, QueryComplexity
        
        router = QueryRouter()
        
        # Simple definition questions
        decision = router.classify_query("What is an altimeter?")
        assert decision.complexity == QueryComplexity.SIMPLE
        
        decision = router.classify_query("Define VOR")
        assert decision.complexity == QueryComplexity.SIMPLE
        
        decision = router.classify_query("What does DME stand for?")
        assert decision.complexity == QueryComplexity.SIMPLE
    
    def test_complex_query_classification(self):
        """Test classification of complex queries."""
        from app.rag.router import QueryRouter, QueryComplexity
        
        router = QueryRouter()
        
        # Complex comparison question
        decision = router.classify_query(
            "Compare ILS and VOR approaches in terms of precision and weather minimums"
        )
        assert decision.complexity == QueryComplexity.COMPLEX
        
        # Multi-domain question
        decision = router.classify_query(
            "How does weather affect fuel consumption during navigation planning?"
        )
        assert decision.complexity == QueryComplexity.COMPLEX
        
        # Trade-off question
        decision = router.classify_query(
            "What are the trade-offs between climbing to avoid turbulence vs deviating around it?"
        )
        assert decision.complexity == QueryComplexity.COMPLEX
    
    def test_moderate_query_classification(self):
        """Test classification of moderate queries."""
        from app.rag.router import QueryRouter, QueryComplexity
        
        router = QueryRouter()
        
        # Applied/procedural question
        decision = router.classify_query(
            "How do you calculate density altitude at a high elevation airport?"
        )
        assert decision.complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
    
    def test_routing_decision_properties(self):
        """Test that routing decisions have proper properties."""
        from app.rag.router import QueryRouter
        
        router = QueryRouter()
        
        decision = router.classify_query("What is true airspeed?")
        
        # Simple queries should have lower top_k
        assert decision.retrieval_top_k <= 10
        assert isinstance(decision.confidence_threshold, float)
        assert 0 < decision.confidence_threshold < 1
        assert isinstance(decision.reasoning, str)
    
    def test_confidence_evaluation(self):
        """Test confidence evaluation."""
        from app.rag.router import QueryRouter, ConfidenceLevel
        
        router = QueryRouter()
        
        # High confidence scenario
        result = router.evaluate_confidence(
            question="What is an altimeter?",
            answer="An altimeter is an instrument that measures altitude.",
            model_confidence=0.9,
            retrieval_scores=[0.85, 0.80, 0.75],
            is_grounded=True,
        )
        assert result.level == ConfidenceLevel.HIGH
        assert result.action == "answer"
        
        # Low confidence scenario
        result = router.evaluate_confidence(
            question="What is something obscure?",
            answer="Some uncertain answer.",
            model_confidence=0.3,
            retrieval_scores=[0.2, 0.15, 0.1],
            is_grounded=False,
        )
        assert result.level == ConfidenceLevel.LOW
        assert result.action in ["refuse", "clarify"]
    
    def test_refusal_detection(self):
        """Test that refusals are properly handled."""
        from app.rag.router import QueryRouter, ConfidenceLevel
        
        router = QueryRouter()
        
        result = router.evaluate_confidence(
            question="What is XYZ?",
            answer="This information is not available in the provided document(s).",
            model_confidence=1.0,
            retrieval_scores=[],
            is_grounded=True,
        )
        assert result.action == "refuse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
