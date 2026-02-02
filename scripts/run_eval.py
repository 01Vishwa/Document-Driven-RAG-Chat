#!/usr/bin/env python
"""
Aviation RAG Chat - Evaluation Script
Run comprehensive evaluation of the RAG system
"""

import sys
import argparse
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

from loguru import logger
from app.eval.evaluate import Evaluator
from app.rag.ingest import rag_engine


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Aviation RAG Chat system"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="./app/eval/questions/evaluation_questions.json",
        help="Path to evaluation questions JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed",
        help="Directory for output files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["factual", "applied", "reasoning", "all"],
        default="all",
        help="Type of questions to evaluate"
    )
    
    args = parser.parse_args()
    
    # Check if documents are indexed
    if rag_engine.vector_store.total_vectors == 0:
        logger.error("No documents indexed! Please run ingestion first.")
        logger.info("Run: python scripts/run_ingest.py")
        sys.exit(1)
    
    logger.info(f"Vector store has {rag_engine.vector_store.total_vectors} vectors")
    
    # Initialize evaluator
    evaluator = Evaluator(questions_file=args.questions)
    
    # Load questions
    questions = evaluator.load_questions()
    
    if not questions:
        logger.error(f"No questions found in {args.questions}")
        sys.exit(1)
    
    # Filter by type if specified
    if args.type != "all":
        questions = [q for q in questions if q.question_type == args.type]
        logger.info(f"Filtered to {len(questions)} {args.type} questions")
    
    # Limit if specified
    if args.limit:
        questions = questions[:args.limit]
        logger.info(f"Limited to {len(questions)} questions")
    
    # Run evaluation
    logger.info(f"Starting evaluation of {len(questions)} questions...")
    
    report = evaluator.run_evaluation(
        questions=questions,
        save_results=True,
        output_dir=args.output,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Questions:      {report.total_questions}")
    print(f"Retrieval Hit Rate:   {report.retrieval_hit_rate:.1%}")
    print(f"Faithfulness Rate:    {report.faithfulness_rate:.1%}")
    print(f"Hallucination Rate:   {report.hallucination_rate:.1%}")
    print(f"Average Confidence:   {report.average_confidence:.1%}")
    print("=" * 60)
    
    print("\nQuestions by Type:")
    for q_type, count in report.by_type.items():
        print(f"  {q_type.capitalize()}: {count}")
    
    print(f"\nResults saved to: {args.output}")
    
    return report


if __name__ == "__main__":
    main()
