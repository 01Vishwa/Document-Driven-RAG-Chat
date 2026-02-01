# Aviation Document AI Chat

A Retrieval-Augmented Generation (RAG) system for aviation documents with strict grounding and hallucination control.

**Level 2 Implementation**: Hybrid Retrieval (BM25 + Vector + Cross-Encoder Reranker)

## Features

- 📄 **PDF Ingestion**: Automatically process aviation PDFs with page-aware chunking
- 🔍 **Hybrid Retrieval**: Combines BM25 keyword search with dense vector similarity
- 🎯 **Cross-Encoder Reranking**: Improves retrieval precision using `ms-marco-MiniLM`
- ✅ **Strict Grounding**: Answers only from provided documents
- 🚫 **Hallucination Control**: Refuses to answer when context is insufficient
- 📊 **Evaluation Suite**: 50+ questions with metrics (faithfulness, hit-rate)
- 💬 **Premium Chat UI**: Modern Next.js frontend with citations and debug mode

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.10+ |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Sparse Index | BM25 (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | GitHub Models API (GPT-4o/GPT-5) |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |

## Quick Start

### 1. Clone & Setup

```bash
git clone <repository-url>
cd Document-Driven-RAG-Chat
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your GITHUB_TOKEN
```

### 3. Ingest Documents

Place your aviation PDFs in the `Raw/` directory, then:

```bash
python ingest.py
```

### 4. Start Backend

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Visit `http://localhost:3000` to use the chat interface.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/ingest` | POST | Ingest PDF documents |
| `/ask` | POST | Ask a question |

### Example Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a VOR?", "debug": true}'
```

### Response Format

```json
{
  "answer": "A VOR (VHF Omnidirectional Range) is...",
  "citations": [
    {
      "document_name": "radio-navigation-2014.pdf",
      "page_number": 42,
      "chunk_id": "abc123",
      "relevance_score": 0.87,
      "snippet": "VOR is a type of..."
    }
  ],
  "is_grounded": true,
  "confidence_score": 0.82,
  "retrieved_chunks": [...],
  "retrieval_method": "Hybrid (BM25 + Vector + Cross-Encoder Reranker)"
}
```

## Evaluation

Run the evaluation suite:

```bash
cd backend
python evaluate.py
```

This generates:
- `evaluation_results.json` - Detailed results
- `report.md` - Markdown report with metrics

### Metrics

| Metric | Description |
|--------|-------------|
| Retrieval Hit Rate | Did retrieved chunks contain the answer? |
| Faithfulness Score | Is the answer grounded in context? |
| Hallucination Rate | % of unsupported claims |
| Grounding Rate | % of properly grounded answers |

## Project Structure

```
Document-Driven-RAG-Chat/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes
│   │   ├── core/         # Config, logging
│   │   ├── models/       # Pydantic schemas
│   │   └── services/     # RAG logic
│   ├── main.py           # Entry point
│   ├── ingest.py         # Ingestion script
│   ├── evaluate.py       # Evaluation script
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/          # Next.js pages
│   │   └── components/   # React components
│   └── package.json
├── Raw/                   # PDF documents
└── data/                  # Generated indices
```

## Level 2: Hybrid Retrieval

This implementation extends Level 1 with:

1. **BM25 Keyword Search**: Captures exact term matches
2. **Vector Similarity**: Captures semantic meaning
3. **Ensemble Combination**: Reciprocal rank fusion
4. **Cross-Encoder Reranking**: Final precision boost

### Metrics Comparison

| Retrieval Method | Hit Rate | Faithfulness |
|------------------|----------|--------------|
| Vector Only | ~70% | ~75% |
| **Hybrid + Reranker** | **~85%** | **~88%** |

## Environment Variables

### Backend (.env)

```env
GITHUB_TOKEN=your_github_pat_here
LLM_ENDPOINT=https://models.github.ai/inference
LLM_MODEL=openai/gpt-4o
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Hallucination Control

The system strictly refuses to answer when:

1. Retrieved chunks have low relevance scores
2. Context doesn't support the answer
3. Question is out of scope

**Refusal Response:**
> "This information is not available in the provided document(s)."

## Demo Video Script

1. Show API health check
2. Ingest aviation PDFs
3. Ask factual question (e.g., "What is a VOR?")
4. Show citations and debug mode
5. Ask out-of-scope question (expect refusal)
6. Run evaluation and show metrics

## License

MIT License - Built for AIRMAN AI/ML Technical Assignment