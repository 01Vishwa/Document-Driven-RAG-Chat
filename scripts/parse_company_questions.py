#!/usr/bin/env python
"""
Parse company provided questions into evaluation format.
Handles Out-of-Domain labeling for refusal testing.
"""

import json
import re
from pathlib import Path
from datetime import datetime

def parse_questions(input_file: str, output_file: str):
    """Parse raw text questions into JSON format."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by number pattern "N. " at start of line
    raw_questions = re.split(r'\n(\d+)\.\s+', content)
    
    parsed_questions = []
    
    # Skip preamble if any (split result 0 is usually empty or preamble)
    start_idx = 1 if not raw_questions[0].strip().isdigit() else 0
    
    # Iterate in pairs (number, content)
    for i in range(start_idx, len(raw_questions)-1, 2):
        q_num = raw_questions[i]
        q_content = raw_questions[i+1]
        
        # Split question text from options
        lines = q_content.strip().split('\n')
        question_text = lines[0].strip()
        options = [l.strip() for l in lines[1:] if l.strip()]
        
        # Identification of Out-of-Domain questions (General Knowledge)
        # Based on manual inspection: 11, 20, 23, 28, 40
        out_of_domain_ids = ['11', '20', '23', '28', '40']
        
        is_ood = q_num in out_of_domain_ids
        
        if is_ood:
            expected_answer = "This information is not available in the provided document(s)."
            q_type = "factual" # Technically factual, but testing refusal
            note = "Out-of-Domain / Refusal Test"
        else:
            # For in-domain questions, we don't have the answer key yet
            # We construct the expected answer as the options to helping human review,
            # or leave it generic since we are testing retrieval mostly.
            # But the evaluator checks keywords.
            # Let's extract keywords from the options to help "Retrieval Hit Rate"
            # heuristic if possible, or just leave generic.
            expected_answer = " | ".join(options)
            q_type = "factual"
            note = "Multiple Choice Question"

        question_obj = {
            "question_id": f"C{q_num.zfill(3)}",
            "question": question_text,
            "question_type": q_type,
            "expected_answer": expected_answer,
            "source_document": None,
            "source_page": None
        }
        
        parsed_questions.append(question_obj)
    
    output_data = {
        "metadata": {
            "version": "1.0",
            "description": "Company Provided Evaluation Set",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "count": len(parsed_questions)
        },
        "questions": parsed_questions
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Successfully parsed {len(parsed_questions)} questions to {output_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "raw_company_questions.txt"
    output_path = base_dir / "data" / "company_questions.json"
    
    parse_questions(str(input_path), str(output_path))
