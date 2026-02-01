"""
Aviation RAG Chat - Evaluation System
Comprehensive evaluation of RAG system quality
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

from app.core.config import settings
from app.core.schemas import (
    QuestionType,
    EvaluationQuestion,
    EvaluationResult,
    EvaluationReport,
    Citation,
)
from app.rag.ingest import rag_engine


@dataclass
class QuestionWithAnswer:
    """Question paired with expected answer for evaluation."""
    question_id: str
    question: str
    question_type: str
    expected_answer: Optional[str] = None
    source_document: Optional[str] = None
    source_page: Optional[int] = None


class Evaluator:
    """
    Evaluate RAG system quality across multiple dimensions:
    - Retrieval hit-rate
    - Faithfulness to sources
    - Hallucination detection
    - Answer quality
    """
    
    def __init__(self, questions_file: str = None):
        """
        Initialize evaluator.
        
        Args:
            questions_file: Path to JSON file with evaluation questions
        """
        self.questions_file = questions_file
        self.questions: List[QuestionWithAnswer] = []
        self.results: List[EvaluationResult] = []
    
    def load_questions(self, questions_file: str = None) -> List[QuestionWithAnswer]:
        """Load questions from JSON file."""
        file_path = questions_file or self.questions_file
        
        if file_path and Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.questions = [
                QuestionWithAnswer(**q) for q in data.get("questions", [])
            ]
            logger.info(f"Loaded {len(self.questions)} questions from {file_path}")
        else:
            logger.warning(f"Questions file not found: {file_path}")
            self.questions = []
        
        return self.questions
    
    def evaluate_single(
        self,
        question: QuestionWithAnswer,
        include_debug: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a single question.
        
        Args:
            question: Question to evaluate
            include_debug: Whether to include retrieved chunks
            
        Returns:
            EvaluationResult with metrics
        """
        logger.debug(f"Evaluating: {question.question[:50]}...")
        
        # Get answer from RAG system
        response = rag_engine.ask(
            question=question.question,
            debug=include_debug,
            top_k=10,
        )
        
        # Analyze retrieval hit
        retrieval_hit = self._check_retrieval_hit(
            question=question,
            chunks=response.retrieved_chunks or [],
        )
        
        # Check faithfulness
        is_faithful = self._check_faithfulness(
            answer=response.answer,
            chunks=response.retrieved_chunks or [],
        )
        
        # Check for hallucination
        is_hallucination = self._check_hallucination(
            answer=response.answer,
            is_grounded=response.is_grounded,
            confidence=response.confidence,
        )
        
        return EvaluationResult(
            question_id=question.question_id,
            question=question.question,
            question_type=QuestionType(question.question_type),
            generated_answer=response.answer,
            expected_answer=question.expected_answer,
            is_grounded=response.is_grounded,
            retrieval_hit=retrieval_hit,
            is_faithful=is_faithful,
            is_hallucination=is_hallucination,
            confidence=response.confidence,
            citations=response.citations,
        )
    
    def _check_retrieval_hit(
        self,
        question: QuestionWithAnswer,
        chunks: List,
    ) -> bool:
        """
        Check if retrieved chunks likely contain the answer.
        
        Uses multiple heuristics:
        - Expected answer keywords in chunks
        - Source document match
        - Page number match
        """
        if not chunks:
            return False
        
        # Combine all chunk content
        chunk_text = " ".join(c.content.lower() for c in chunks)
        
        # Check for expected answer keywords
        if question.expected_answer:
            keywords = self._extract_keywords(question.expected_answer)
            keyword_matches = sum(1 for k in keywords if k.lower() in chunk_text)
            if keyword_matches >= len(keywords) * 0.3:  # 30% keyword match
                return True
        
        # Check source document match
        if question.source_document:
            for chunk in chunks:
                if question.source_document.lower() in chunk.source_file.lower():
                    # Additional check for page if specified
                    if question.source_page:
                        if abs(chunk.page_number - question.source_page) <= 2:
                            return True
                    else:
                        return True
        
        # Check question keywords in chunks
        question_keywords = self._extract_keywords(question.question)
        keyword_matches = sum(1 for k in question_keywords if k.lower() in chunk_text)
        
        return keyword_matches >= len(question_keywords) * 0.5
    
    def _check_faithfulness(
        self,
        answer: str,
        chunks: List,
    ) -> bool:
        """
        Check if answer is faithful to retrieved chunks.
        
        An answer is faithful if its claims are supported by the chunks.
        """
        # Refusal is always faithful (correct behavior)
        refusal_phrases = [
            "not available in the provided document",
            "cannot find this information",
            "not mentioned",
        ]
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            return True
        
        if not chunks:
            return False
        
        # Combine chunk content
        chunk_text = " ".join(c.content.lower() for c in chunks)
        
        # Extract key facts from answer
        answer_sentences = answer.split('.')
        
        # Check if key sentences have support in chunks
        supported_sentences = 0
        for sentence in answer_sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                supported_sentences += 1
                continue
            
            # Check for keyword overlap
            sentence_words = set(sentence.lower().split())
            chunk_words = set(chunk_text.split())
            
            overlap = len(sentence_words & chunk_words) / max(len(sentence_words), 1)
            if overlap > 0.3:  # 30% word overlap
                supported_sentences += 1
        
        faithfulness_ratio = supported_sentences / max(len(answer_sentences), 1)
        return faithfulness_ratio >= 0.7  # 70% of sentences supported
    
    def _check_hallucination(
        self,
        answer: str,
        is_grounded: bool,
        confidence: float,
    ) -> bool:
        """
        Check if answer contains hallucination.
        
        Hallucination indicators:
        - Low confidence
        - Not grounded
        - Contains speculative language
        - Contains information not in typical aviation docs
        """
        # Refusal is not hallucination
        if "not available in the provided document" in answer.lower():
            return False
        
        # Check grounding and confidence
        if not is_grounded and confidence < 0.5:
            return True
        
        # Check for speculative language
        speculative_phrases = [
            "i think",
            "probably",
            "might be",
            "possibly",
            "it seems",
            "i believe",
            "generally speaking",
            "in my opinion",
        ]
        
        if any(phrase in answer.lower() for phrase in speculative_phrases):
            return True
        
        return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        import re
        
        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "what", "which", "who", "whom", "this", "that", "these",
            "those", "am", "and", "but", "if", "or", "because", "until",
            "while", "how", "where", "when", "why",
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter and return
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:20]  # Limit to top 20
    
    def run_evaluation(
        self,
        questions: List[QuestionWithAnswer] = None,
        save_results: bool = True,
        output_dir: str = None,
    ) -> EvaluationReport:
        """
        Run full evaluation on all questions.
        
        Args:
            questions: Questions to evaluate (uses loaded questions if None)
            save_results: Whether to save results to file
            output_dir: Directory for output files
            
        Returns:
            EvaluationReport with all metrics
        """
        questions = questions or self.questions
        
        if not questions:
            raise ValueError("No questions to evaluate")
        
        logger.info(f"Starting evaluation of {len(questions)} questions")
        
        self.results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}")
            
            try:
                result = self.evaluate_single(question)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating question {question.question_id}: {e}")
                # Add failed result
                self.results.append(EvaluationResult(
                    question_id=question.question_id,
                    question=question.question,
                    question_type=QuestionType(question.question_type),
                    generated_answer=f"Error: {e}",
                    expected_answer=question.expected_answer,
                    is_grounded=False,
                    retrieval_hit=False,
                    is_faithful=False,
                    is_hallucination=True,
                    confidence=0.0,
                    citations=[],
                    notes=str(e),
                ))
        
        # Generate report
        report = self._generate_report()
        
        # Save results
        if save_results:
            output_path = Path(output_dir) if output_dir else settings.processed_dir
            self._save_results(report, output_path)
        
        return report
    
    def _generate_report(self) -> EvaluationReport:
        """Generate evaluation report from results."""
        total = len(self.results)
        
        if total == 0:
            return EvaluationReport(
                total_questions=0,
                by_type={},
                retrieval_hit_rate=0.0,
                faithfulness_rate=0.0,
                hallucination_rate=0.0,
                average_confidence=0.0,
                results=[],
                best_answers=[],
                worst_answers=[],
            )
        
        # Count by type
        by_type = {}
        for result in self.results:
            q_type = result.question_type.value
            by_type[q_type] = by_type.get(q_type, 0) + 1
        
        # Calculate metrics
        retrieval_hits = sum(1 for r in self.results if r.retrieval_hit)
        faithful_count = sum(1 for r in self.results if r.is_faithful)
        hallucination_count = sum(1 for r in self.results if r.is_hallucination)
        total_confidence = sum(r.confidence for r in self.results)
        
        # Sort by confidence for best/worst
        sorted_results = sorted(self.results, key=lambda r: r.confidence, reverse=True)
        
        # Filter for meaningful best/worst (exclude refusals from best)
        answerable = [
            r for r in sorted_results 
            if "not available" not in r.generated_answer.lower()
        ]
        
        best_answers = answerable[:5] if answerable else sorted_results[:5]
        worst_answers = sorted_results[-5:] if len(sorted_results) >= 5 else sorted_results
        
        return EvaluationReport(
            total_questions=total,
            by_type=by_type,
            retrieval_hit_rate=retrieval_hits / total,
            faithfulness_rate=faithful_count / total,
            hallucination_rate=hallucination_count / total,
            average_confidence=total_confidence / total,
            results=self.results,
            best_answers=best_answers,
            worst_answers=worst_answers,
        )
    
    def _save_results(self, report: EvaluationReport, output_dir: Path):
        """Save evaluation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        results_file = output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "total_questions": report.total_questions,
                    "by_type": report.by_type,
                    "retrieval_hit_rate": report.retrieval_hit_rate,
                    "faithfulness_rate": report.faithfulness_rate,
                    "hallucination_rate": report.hallucination_rate,
                    "average_confidence": report.average_confidence,
                    "results": [
                        {
                            "question_id": r.question_id,
                            "question": r.question,
                            "question_type": r.question_type.value,
                            "generated_answer": r.generated_answer,
                            "expected_answer": r.expected_answer,
                            "is_grounded": r.is_grounded,
                            "retrieval_hit": r.retrieval_hit,
                            "is_faithful": r.is_faithful,
                            "is_hallucination": r.is_hallucination,
                            "confidence": r.confidence,
                        }
                        for r in report.results
                    ],
                },
                f,
                indent=2,
            )
        
        logger.info(f"Saved evaluation results to {results_file}")
        
        # Generate markdown report
        report_file = output_dir / f"report_{timestamp}.md"
        self._generate_markdown_report(report, report_file)
        
        logger.info(f"Saved markdown report to {report_file}")
    
    def _generate_markdown_report(self, report: EvaluationReport, output_file: Path):
        """Generate human-readable markdown report."""
        md_content = f"""# Aviation RAG Chat - Evaluation Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

| Metric | Value |
|--------|-------|
| Total Questions | {report.total_questions} |
| Retrieval Hit Rate | {report.retrieval_hit_rate:.1%} |
| Faithfulness Rate | {report.faithfulness_rate:.1%} |
| Hallucination Rate | {report.hallucination_rate:.1%} |
| Average Confidence | {report.average_confidence:.1%} |

## Questions by Type

| Type | Count |
|------|-------|
"""
        
        for q_type, count in report.by_type.items():
            md_content += f"| {q_type.capitalize()} | {count} |\n"
        
        md_content += """
## Best Answers (Top 5)

"""
        
        for i, result in enumerate(report.best_answers, 1):
            md_content += f"""### {i}. {result.question}

**Answer:** {result.generated_answer[:500]}{"..." if len(result.generated_answer) > 500 else ""}

- **Confidence:** {result.confidence:.1%}
- **Grounded:** {"Yes" if result.is_grounded else "No"}
- **Faithful:** {"Yes" if result.is_faithful else "No"}
- **Type:** {result.question_type.value}

---

"""
        
        md_content += """
## Worst Answers (Bottom 5)

"""
        
        for i, result in enumerate(report.worst_answers, 1):
            md_content += f"""### {i}. {result.question}

**Answer:** {result.generated_answer[:500]}{"..." if len(result.generated_answer) > 500 else ""}

- **Confidence:** {result.confidence:.1%}
- **Grounded:** {"Yes" if result.is_grounded else "No"}
- **Faithful:** {"Yes" if result.is_faithful else "No"}
- **Hallucination:** {"Yes" if result.is_hallucination else "No"}
- **Type:** {result.question_type.value}
{f"- **Notes:** {result.notes}" if result.notes else ""}

---

"""
        
        md_content += """
## Analysis

### Strengths
- Questions with high retrieval hit rate indicate good embedding and indexing quality
- High faithfulness rate shows the grounding mechanisms are working

### Areas for Improvement
- Questions with low retrieval hit may need better chunking or embedding strategies
- Hallucinations should be investigated for patterns

### Recommendations
1. Review worst-performing questions for retrieval improvements
2. Analyze hallucination patterns for additional guardrails
3. Consider hybrid retrieval (BM25 + vector) for keyword-heavy questions

"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)


# Singleton instance
evaluator = Evaluator()
