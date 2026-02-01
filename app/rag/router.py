"""
Aviation RAG Chat - Query Router & Confidence Thresholding
Routes queries to appropriate models based on complexity and handles low-confidence responses
"""

import re
from enum import Enum
from typing import Tuple, Optional, List
from dataclasses import dataclass
from loguru import logger

from app.core.config import settings


class QueryComplexity(str, Enum):
    """Query complexity levels for routing."""
    SIMPLE = "simple"      # Direct lookups, definitions
    MODERATE = "moderate"  # Applied questions, single-step reasoning
    COMPLEX = "complex"    # Multi-step reasoning, trade-offs, conditionals


class ConfidenceLevel(str, Enum):
    """Confidence levels for response quality."""
    HIGH = "high"          # >= 0.8 - Answer confidently
    MEDIUM = "medium"      # 0.5 - 0.8 - Answer with caveats
    LOW = "low"            # < 0.5 - Refuse or ask clarification


@dataclass
class RoutingDecision:
    """Result of query routing analysis."""
    complexity: QueryComplexity
    confidence_threshold: float
    recommended_model: str
    reasoning: str
    should_use_reranker: bool
    retrieval_top_k: int


@dataclass
class ConfidenceResult:
    """Result of confidence thresholding."""
    level: ConfidenceLevel
    score: float
    action: str  # "answer", "answer_with_caveat", "refuse", "clarify"
    clarification_question: Optional[str] = None


class QueryRouter:
    """
    Routes queries to appropriate processing paths based on complexity.
    
    Routing Strategy:
    =================
    
    1. **Simple Questions** (definitions, direct lookups):
       - Use faster/smaller model if available
       - Lower retrieval top_k (fewer chunks needed)
       - Skip reranker for speed
       - Examples: "What is VOR?", "Define altimeter"
    
    2. **Moderate Questions** (applied, procedural):
       - Use standard model
       - Standard retrieval with reranking
       - Examples: "How to calculate density altitude?"
    
    3. **Complex Questions** (multi-step reasoning, trade-offs):
       - Use strongest model
       - Higher retrieval top_k
       - Always use reranker
       - May require multiple retrieval passes
       - Examples: "Compare ILS vs VOR approach in low visibility with fuel constraints"
    
    Confidence Thresholding:
    ========================
    
    After answer generation, evaluate confidence:
    - HIGH (>= 0.8): Return answer directly
    - MEDIUM (0.5-0.8): Return answer with confidence caveat
    - LOW (< 0.5): Either refuse or ask clarification question
    
    Clarification triggers:
    - Ambiguous question detected
    - Multiple possible interpretations
    - Missing context that user could provide
    """
    
    # Keywords/patterns indicating simple questions
    SIMPLE_PATTERNS = [
        r'^what is (a |an |the )?',
        r'^define\b',
        r'^what does .+ stand for',
        r'^what is the definition',
        r'^what is the meaning',
        r'^what is the purpose of',
        r'^what are the types of',
        r'^list the',
        r'^name the',
        r'^what is the abbreviation',
    ]
    
    # Keywords/patterns indicating complex questions
    COMPLEX_PATTERNS = [
        r'\bcompare\b',
        r'\bcontrast\b',
        r'\bwhat if\b',
        r'\bhow would\b',
        r'\bshould .+ or\b',
        r'\btrade-?off',
        r'\bweigh .+ against\b',
        r'\bbest .+ when\b',
        r'\boptimal\b',
        r'\bmultiple\b.*\bfactors\b',
        r'\bconsidering\b.*\band\b.*\band\b',
        r'\bgiven that\b.*\bwhat\b',
        r'\bif .+ then .+ but\b',
        r'\bbalance\b.*\bbetween\b',
    ]
    
    # Keywords indicating ambiguity (may need clarification)
    AMBIGUOUS_PATTERNS = [
        r'^(it|this|that|these|those)\b',  # Unclear reference
        r'\bthe (plane|aircraft|pilot)\b(?!.*specific)',  # Generic reference
        r'\b(best|optimal|recommended)\b(?!.*for)',  # Subjective without context
    ]
    
    # Aviation-specific complexity indicators
    MULTI_DOMAIN_KEYWORDS = [
        ('weather', 'navigation'),
        ('fuel', 'weather'),
        ('weight', 'performance'),
        ('emergency', 'procedure'),
        ('regulation', 'operational'),
    ]
    
    def __init__(self):
        """Initialize the query router."""
        self.simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
        self.complex_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_PATTERNS]
        self.ambiguous_patterns = [re.compile(p, re.IGNORECASE) for p in self.AMBIGUOUS_PATTERNS]
        
        logger.info("QueryRouter initialized")
    
    def _count_pattern_matches(self, text: str, patterns: list) -> int:
        """Count how many patterns match the text."""
        return sum(1 for p in patterns if p.search(text))
    
    def _detect_multi_domain(self, text: str) -> bool:
        """Check if question spans multiple aviation domains."""
        text_lower = text.lower()
        for domain1, domain2 in self.MULTI_DOMAIN_KEYWORDS:
            if domain1 in text_lower and domain2 in text_lower:
                return True
        return False
    
    def _count_clauses(self, text: str) -> int:
        """Estimate sentence complexity by counting clauses."""
        # Count conjunctions and conditional markers
        markers = [
            r'\band\b', r'\bor\b', r'\bbut\b', r'\bif\b', 
            r'\bwhen\b', r'\bwhile\b', r'\balthough\b',
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b',
        ]
        count = sum(len(re.findall(m, text, re.IGNORECASE)) for m in markers)
        return count
    
    def _is_ambiguous(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the question is ambiguous and suggest clarification.
        
        Returns:
            Tuple of (is_ambiguous, clarification_question)
        """
        for pattern in self.ambiguous_patterns:
            if pattern.search(text):
                # Generate clarification question based on the ambiguity
                if re.search(r'^(it|this|that)\b', text, re.IGNORECASE):
                    return True, "Could you please specify what you're referring to? Your question starts with a pronoun without clear context."
                if re.search(r'\b(best|optimal)\b', text, re.IGNORECASE):
                    return True, "Could you provide more context about the specific conditions or constraints? This will help me give a more accurate answer."
        
        return False, None
    
    def classify_query(self, question: str) -> RoutingDecision:
        """
        Classify a query and determine routing.
        
        Args:
            question: The user's question
            
        Returns:
            RoutingDecision with recommended processing path
        """
        question = question.strip()
        
        # Count pattern matches
        simple_score = self._count_pattern_matches(question, self.simple_patterns)
        complex_score = self._count_pattern_matches(question, self.complex_patterns)
        
        # Check multi-domain
        is_multi_domain = self._detect_multi_domain(question)
        
        # Count clauses for complexity estimation
        clause_count = self._count_clauses(question)
        
        # Question length as a factor
        word_count = len(question.split())
        
        # Determine complexity
        if simple_score > 0 and complex_score == 0 and word_count < 15:
            complexity = QueryComplexity.SIMPLE
            reasoning = f"Simple pattern matched, short question ({word_count} words)"
        elif complex_score > 0 or is_multi_domain or clause_count >= 3:
            complexity = QueryComplexity.COMPLEX
            reasons = []
            if complex_score > 0:
                reasons.append("complex pattern detected")
            if is_multi_domain:
                reasons.append("spans multiple domains")
            if clause_count >= 3:
                reasons.append(f"high clause count ({clause_count})")
            reasoning = f"Complex: {', '.join(reasons)}"
        else:
            complexity = QueryComplexity.MODERATE
            reasoning = "Standard applied/procedural question"
        
        # Determine routing based on complexity
        if complexity == QueryComplexity.SIMPLE:
            return RoutingDecision(
                complexity=complexity,
                confidence_threshold=0.6,  # Lower threshold for simple facts
                recommended_model=settings.llm_model,  # Could use smaller model
                reasoning=reasoning,
                should_use_reranker=False,  # Skip for speed
                retrieval_top_k=5,  # Fewer chunks needed
            )
        elif complexity == QueryComplexity.COMPLEX:
            return RoutingDecision(
                complexity=complexity,
                confidence_threshold=0.7,  # Higher threshold for complex
                recommended_model=settings.llm_model,  # Use strongest model
                reasoning=reasoning,
                should_use_reranker=True,  # Always rerank
                retrieval_top_k=15,  # More context needed
            )
        else:  # MODERATE
            return RoutingDecision(
                complexity=complexity,
                confidence_threshold=0.65,
                recommended_model=settings.llm_model,
                reasoning=reasoning,
                should_use_reranker=settings.use_reranker,
                retrieval_top_k=settings.top_k_retrieval,
            )
    
    def evaluate_confidence(
        self,
        question: str,
        answer: str,
        model_confidence: float,
        retrieval_scores: List[float],
        is_grounded: bool,
    ) -> ConfidenceResult:
        """
        Evaluate confidence and determine response action.
        
        Confidence is calculated from:
        1. Model's reported confidence
        2. Retrieval similarity scores
        3. Grounding verification result
        4. Answer characteristics
        
        Args:
            question: Original question
            answer: Generated answer
            model_confidence: Confidence from LLM
            retrieval_scores: Similarity scores from retrieval
            is_grounded: Whether answer passed grounding check
            
        Returns:
            ConfidenceResult with action recommendation
        """
        # Check if this is a refusal (already handled)
        if "not available in the provided document" in answer.lower():
            return ConfidenceResult(
                level=ConfidenceLevel.HIGH,
                score=1.0,
                action="refuse",
            )
        
        # Calculate composite confidence score
        # Weight: model_confidence (40%), retrieval (40%), grounding (20%)
        
        # Retrieval confidence: average of top scores, penalized if low
        if retrieval_scores:
            avg_retrieval = sum(retrieval_scores[:3]) / min(3, len(retrieval_scores))
            retrieval_confidence = min(1.0, avg_retrieval)  # Cap at 1.0
        else:
            retrieval_confidence = 0.0
        
        # Grounding factor
        grounding_factor = 1.0 if is_grounded else 0.5
        
        # Composite score
        composite_score = (
            0.4 * model_confidence +
            0.4 * retrieval_confidence +
            0.2 * grounding_factor
        )
        
        # Check for ambiguity
        is_ambiguous, clarification = self._is_ambiguous(question)
        
        # Determine confidence level and action
        if composite_score >= 0.8:
            return ConfidenceResult(
                level=ConfidenceLevel.HIGH,
                score=composite_score,
                action="answer",
            )
        elif composite_score >= 0.5:
            # Medium confidence - answer with caveat
            return ConfidenceResult(
                level=ConfidenceLevel.MEDIUM,
                score=composite_score,
                action="answer_with_caveat",
            )
        else:
            # Low confidence - refuse or ask clarification
            if is_ambiguous and clarification:
                return ConfidenceResult(
                    level=ConfidenceLevel.LOW,
                    score=composite_score,
                    action="clarify",
                    clarification_question=clarification,
                )
            else:
                return ConfidenceResult(
                    level=ConfidenceLevel.LOW,
                    score=composite_score,
                    action="refuse",
                )
    
    def format_response_with_confidence(
        self,
        answer: str,
        confidence_result: ConfidenceResult,
    ) -> str:
        """
        Format the final response based on confidence evaluation.
        
        Args:
            answer: The generated answer
            confidence_result: Result from confidence evaluation
            
        Returns:
            Formatted response string
        """
        if confidence_result.action == "answer":
            return answer
        
        elif confidence_result.action == "answer_with_caveat":
            caveat = (
                f"\n\n**Note**: This answer is provided with moderate confidence "
                f"({confidence_result.score:.0%}). Please verify critical information "
                f"with official sources."
            )
            return answer + caveat
        
        elif confidence_result.action == "clarify":
            return (
                f"I need some clarification to provide an accurate answer.\n\n"
                f"{confidence_result.clarification_question}\n\n"
                f"Once you provide more details, I can give you a more precise response "
                f"from the aviation documents."
            )
        
        else:  # refuse
            return "This information is not available in the provided document(s)."


# Singleton instance
query_router = QueryRouter()
