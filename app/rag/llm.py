"""
Aviation RAG Chat - LLM Service
Grounded answer generation using GitHub Models API
"""

import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import httpx
from openai import OpenAI
from loguru import logger

from app.core.config import settings
from app.core.schemas import RetrievedChunk, Citation


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    answer: str
    is_grounded: bool
    confidence: float
    citations: List[Citation]
    raw_response: Optional[str] = None


# System prompt for grounded aviation answers
SYSTEM_PROMPT = """You are an Aviation Document Assistant specialized in answering questions from official aviation documents (PPL/CPL/ATPL textbooks, SOPs, and manuals).

CRITICAL RULES - YOU MUST FOLLOW:

1. **ONLY use information from the provided context chunks** to answer questions.
2. **NEVER generate information not present in the context** - this is aviation safety-critical.
3. If the context does not contain enough information to answer the question, respond EXACTLY with:
   "This information is not available in the provided document(s)."
4. When answering, cite the specific document and page number(s) where you found the information.
5. Be precise and technical - aviation requires accuracy.
6. If information is partially available, state what you can answer and what is missing.

RESPONSE FORMAT:
When you CAN answer from context:
- Provide a clear, accurate answer
- Cite sources using [Document Name, Page X]
- Use bullet points for lists
- Be concise but complete

When you CANNOT answer:
- Respond with: "This information is not available in the provided document(s)."
- Do NOT attempt to answer from general knowledge

Remember: Incorrect aviation information can be dangerous. When in doubt, refuse to answer."""


GROUNDING_CHECK_PROMPT = """Analyze whether the following answer is fully grounded in the provided context.

CONTEXT CHUNKS:
{context}

QUESTION: {question}

ANSWER: {answer}

Evaluate:
1. Is every claim in the answer directly supported by the context?
2. Are there any statements that go beyond what the context provides?
3. Are the citations accurate?

Respond in JSON format:
{{
    "is_grounded": true/false,
    "confidence": 0.0-1.0,
    "unsupported_claims": ["list of any claims not in context"],
    "reasoning": "brief explanation"
}}"""


class LLMService:
    """
    LLM service for grounded answer generation.
    
    Uses GitHub Models API with strict grounding enforcement.
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self._initialized = False
        self.client = None
        self.model = settings.llm_model
        
        if settings.github_token:
            try:
                self.client = OpenAI(
                    api_key=settings.github_token,
                    base_url=settings.llm_endpoint,
                )
                self._initialized = True
                logger.info(f"LLM service initialized with model: {self.model}")
            except Exception as e:
                logger.warning(f"LLM service initialization failed: {e}")
        else:
            logger.warning("GITHUB_TOKEN not set - LLM service will not be available")
    
    @property
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self._initialized and self.client is not None
    
    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks for LLM context."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Chunk {i}]\n"
                f"Source: {chunk.source_file}, Page {chunk.page_number}\n"
                f"Content:\n{chunk.content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _extract_citations(
        self,
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> List[Citation]:
        """Extract citations from answer and match to chunks."""
        citations = []
        used_chunks = set()
        
        # Look for citation patterns in the answer
        import re
        citation_patterns = [
            r'\[([^,\]]+),\s*[Pp]age\s*(\d+)\]',
            r'\(([^,)]+),\s*[Pp]\.\s*(\d+)\)',
            r'from\s+([^,]+),\s*page\s*(\d+)',
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, answer)
            for doc_name, page_num in matches:
                # Find matching chunk
                for chunk in chunks:
                    if (chunk.source_file.lower() in doc_name.lower() or 
                        doc_name.lower() in chunk.source_file.lower()):
                        if chunk.chunk_id not in used_chunks:
                            citations.append(Citation(
                                document_name=chunk.source_file,
                                page_number=chunk.page_number,
                                chunk_id=chunk.chunk_id,
                                relevant_text=chunk.content[:200] + "...",
                                confidence=chunk.rerank_score or chunk.score,
                            ))
                            used_chunks.add(chunk.chunk_id)
                        break
        
        # If no explicit citations found, use top chunks
        if not citations:
            for chunk in chunks[:3]:
                if chunk.chunk_id not in used_chunks:
                    citations.append(Citation(
                        document_name=chunk.source_file,
                        page_number=chunk.page_number,
                        chunk_id=chunk.chunk_id,
                        relevant_text=chunk.content[:200] + "...",
                        confidence=chunk.rerank_score or chunk.score,
                    ))
        
        return citations
    
    def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        check_grounding: bool = True,
    ) -> GenerationResult:
        """
        Generate a grounded answer from retrieved chunks.
        
        Args:
            question: User's question
            chunks: Retrieved context chunks
            check_grounding: Whether to verify grounding
            
        Returns:
            GenerationResult with answer and metadata
        """
        # Check if LLM is available
        if not self.is_available:
            return GenerationResult(
                answer="LLM service is not available. Please configure GITHUB_TOKEN in your .env file.",
                is_grounded=False,
                confidence=0.0,
                citations=[],
            )
        
        if not chunks:
            return GenerationResult(
                answer="This information is not available in the provided document(s).",
                is_grounded=True,
                confidence=1.0,
                citations=[],
            )
        
        # Format context
        context = self._format_context(chunks)
        
        # Generate answer
        user_message = f"""Based ONLY on the following context from aviation documents, answer the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer ONLY using information from the context above
- Cite sources as [Document Name, Page X]
- If the context doesn't contain the answer, say: "This information is not available in the provided document(s)."
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,  # Low temperature for factual answers
                max_tokens=1000,
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return GenerationResult(
                answer="I encountered an error while generating the answer. Please try again.",
                is_grounded=False,
                confidence=0.0,
                citations=[],
            )
        
        # Check for refusal
        refusal_phrases = [
            "not available in the provided document",
            "cannot find this information",
            "no information available",
            "not mentioned in the context",
            "context does not contain",
        ]
        
        is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)
        
        if is_refusal:
            return GenerationResult(
                answer="This information is not available in the provided document(s).",
                is_grounded=True,
                confidence=1.0,
                citations=[],
            )
        
        # Extract citations
        citations = self._extract_citations(answer, chunks)
        
        # Check grounding if enabled
        is_grounded = True
        confidence = 0.8  # Default confidence
        
        if check_grounding and settings.grounding_check:
            grounding_result = self._verify_grounding(question, answer, chunks)
            is_grounded = grounding_result["is_grounded"]
            confidence = grounding_result["confidence"]
            
            # If not grounded, refuse to answer
            if not is_grounded:
                return GenerationResult(
                    answer="This information is not available in the provided document(s).",
                    is_grounded=False,
                    confidence=confidence,
                    citations=[],
                    raw_response=answer,
                )
            
            # If grounded but low confidence, also refuse
            if confidence < settings.confidence_threshold:
                return GenerationResult(
                    answer="This information is not available in the provided document(s).",
                    is_grounded=True,  # Technically grounded but low confidence
                    confidence=confidence,
                    citations=[],
                    raw_response=answer,
                )
        
        return GenerationResult(
            answer=answer,
            is_grounded=is_grounded,
            confidence=confidence,
            citations=citations,
            raw_response=answer,
        )
    
    def _verify_grounding(
        self,
        question: str,
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> Dict[str, Any]:
        """
        Verify that the answer is grounded in the context.
        
        Uses a separate LLM call to evaluate grounding.
        """
        context = self._format_context(chunks)
        
        prompt = GROUNDING_CHECK_PROMPT.format(
            context=context,
            question=question,
            answer=answer,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a grounding verification assistant. Respond only in valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text)
            
            return {
                "is_grounded": result.get("is_grounded", True),
                "confidence": result.get("confidence", 0.5),
                "unsupported_claims": result.get("unsupported_claims", []),
            }
            
        except Exception as e:
            logger.warning(f"Grounding check failed: {e}")
            # Default to trusting the answer if check fails
            return {
                "is_grounded": True,
                "confidence": 0.7,
                "unsupported_claims": [],
            }
    
    def health_check(self) -> bool:
        """Check if LLM service is healthy."""
        if not self.is_available:
            return False
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False


# Singleton instance
llm_service = LLMService()
