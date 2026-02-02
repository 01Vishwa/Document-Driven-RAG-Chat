"""
Aviation RAG Chat - Benchmark & Comparison Tool
Compare baseline (vector-only) vs. hybrid retrieval performance (Level 2 Requirement)
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd
from loguru import logger

from app.core.config import settings
from app.api.server import rag_engine
from app.eval.evaluate import Evaluator, QuestionWithAnswer

class Benchmarker:
    def __init__(self, questions_file: str = None):
        self.evaluator = Evaluator(questions_file)
        self.questions = self.evaluator.load_questions()
        
    def run_benchmark(self, output_dir: str = "data/benchmark"):
        """Run comparison between Vector-Only and Hybrid Retrieval."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        logger.info(f"Starting benchmark on {len(self.questions)} questions...")
        
        # 1. Baseline: Vector Only (No BM25, No Reranker)
        logger.info("Running Baseline: Vector-Only Retrieval...")
        settings.use_bm25 = False
        settings.use_reranker = False
        
        baseline_report = self.evaluator.run_evaluation(
            questions=self.questions, 
            save_results=False
        )
        
        # 2. Enhanced: Hybrid + Reranker (Level 2)
        logger.info("Running Enhanced: Hybrid Retrieval + Reranker...")
        settings.use_bm25 = True
        settings.use_reranker = True
        
        enhanced_report = self.evaluator.run_evaluation(
            questions=self.questions, 
            save_results=False
        )
        
        # 3. Compare Metrics
        comparison = {
            "metric": [
                "Retrieval Hit Rate", 
                "Faithfulness", 
                "Hallucination Rate", 
                "Avg Confidence"
            ],
            "baseline_vector_only": [
                baseline_report.retrieval_hit_rate,
                baseline_report.faithfulness_rate,
                baseline_report.hallucination_rate,
                baseline_report.average_confidence
            ],
            "enhanced_hybrid": [
                enhanced_report.retrieval_hit_rate,
                enhanced_report.faithfulness_rate,
                enhanced_report.hallucination_rate,
                enhanced_report.average_confidence
            ]
        }
        
        # Calculate Improvement
        comparison["improvement"] = [
            enhanced - baseline 
            for enhanced, baseline in zip(comparison["enhanced_hybrid"], comparison["baseline_vector_only"])
        ]
        
        # Save Report
        df = pd.DataFrame(comparison)
        csv_path = output_path / "benchmark_comparison.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate Markdown Summary
        md_content = f"""# Retrieval Benchmark: Vector vs. Hybrid

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Questions:** {len(self.questions)}

## Performance Comparison

| Metric | Vector-Only (Baseline) | Hybrid + Reranker (Enhanced) | Improvement |
|--------|------------------------|------------------------------|-------------|
"""
        for i, row in df.iterrows():
            md_content += f"| {row['metric']} | {row['baseline_vector_only']:.2%} | {row['enhanced_hybrid']:.2%} | {row['improvement']:+.2%} |\n"
            
        md_path = output_path / "benchmark_report.md"
        with open(md_path, "w") as f:
            f.write(md_content)
            
        logger.info(f"Benchmark complete. Report saved to {md_path}")
        print(md_content)

if __name__ == "__main__":
    # Ensure we use a valid questions file
    questions_file = "data/company_questions.json"
    # Note: Evaluator expects JSON, so we rely on default path if not specified
    benchmarker = Benchmarker()
    benchmarker.run_benchmark()
