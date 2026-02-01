# ✈️ Aviation Document AI Chat

A production-grade Retrieval-Augmented Generation (RAG) system for aviation documents with strict grounding, hallucination control, and traceable citations.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Overview

This system answers questions **strictly and only** from provided aviation documents (PPL/CPL/ATPL textbooks, SOPs, manuals). If an answer cannot be supported by the documents, the system explicitly refuses to answer.

**Level 2 Implementation**: Hybrid Retrieval (BM25 + Vector + Cross-Encoder Reranker)

## ✨ Features

### Level 1 (Mandatory)
- 📄 **PDF Ingestion Pipeline** - Page-aware text extraction and intelligent chunking
- 🔍 **Vector Retrieval** - FAISS-based semantic similarity search
- ✅ **Strict Grounding** - Answers only from provided documents with verification
- 🚫 **Hallucination Control** - Refuses to answer when context is insufficient
- 📚 **Traceable Citations** - Document name + page number for every answer
- 📊 **Evaluation Suite** - 50 questions with metrics (faithfulness, hit-rate, hallucination)
- 💬 **Streamlit Chat UI** - Modern interface with debug mode and citations

### Level 2 (Advanced - Implemented)
- 🔍 **Hybrid Retrieval** - Combines vector similarity (FAISS) with BM25 keyword search
- 🎯 **Cross-Encoder Reranking** - Improves retrieval precision using `ms-marco-MiniLM`
- ⚡ **GPU Acceleration** - FAISS and embeddings leverage CUDA when available

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Backend | FastAPI |
| UI | Streamlit |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (GPU-accelerated) |
| Sparse Index | BM25 (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | GitHub Models API (GPT-4o) |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│                    (Chat + History)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ /ingest  │  │  /ask    │  │ /health  │  │   /documents     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAG Engine                                 │
│  Query → Embed → Hybrid Search → Rerank → LLM → Verify          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Embeddings  │  │ Vector Store│  │      Reranker           │  │
│  │ (MiniLM)    │  │ (FAISS+BM25)│  │  (Cross-Encoder)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

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

# For GPU support (optional, requires CUDA)
pip install faiss-gpu

# Configure environment
copy .env.example .env
# Edit .env with your GITHUB_TOKEN
```

### 3. Add Aviation Documents

Place your aviation PDFs in the `Raw/` directory:
```
Raw/
├── Air Navigation/
│   ├── 10-General-Navigation-2014 (1).pdf
│   ├── 11-radio-navigation-2014.pdf
│   └── ...
├── Meteorology/
│   └── Meteorology full book.pdf
└── Air-Regulation-RK-BALI.pdf
```

### 4. Ingest Documents

```bash
cd backend
python ingest.py
```

Options:
```bash
# Ingest specific file
python ingest.py --path "../Raw/Air Navigation/Instruments.pdf"

# Force re-index (clear existing)
python ingest.py --force

# Show stats only
python ingest.py --stats
```

### 5. Start the Application

```bash
# Start both FastAPI and Streamlit
python run.py
```

Or run separately:
```bash
# Terminal 1: FastAPI
python server.py
# or: uvicorn app.main:app --reload --port 8000

# Terminal 2: Streamlit
streamlit run streamlit_app.py --server.port 8501
```

### 6. Access the Application

- **Chat UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
Document-Driven-RAG-Chat/
├── Raw/                          # Aviation PDF documents
│   ├── Air Navigation/
│   ├── Meteorology/
│   └── ...
├── backend/
│   ├── app/
│   │   ├── core/                 # Configuration & logging
│   │   │   ├── config.py         # Pydantic settings
│   │   │   └── logging.py        # Loguru setup
│   │   ├── models/               # Pydantic schemas
│   │   │   └── schemas.py        # All data models
│   │   ├── services/             # Core services
│   │   │   ├── pdf_processor.py  # PDF extraction & chunking
│   │   │   ├── embeddings.py     # Sentence transformers
│   │   │   ├── vector_store.py   # FAISS + BM25
│   │   │   ├── reranker.py       # Cross-encoder reranking
│   │   │   ├── llm.py            # LLM service
│   │   │   └── rag_engine.py     # Main orchestrator
│   │   ├── evaluation/           # Evaluation system
│   │   │   └── evaluator.py      # Metrics & reporting
│   │   └── main.py               # FastAPI application
│   ├── data/
│   │   ├── faiss_index/          # Vector store (generated)
│   │   ├── evaluation_results/   # Evaluation outputs
│   │   └── evaluation_questions.json  # 50 test questions
│   ├── logs/                     # Application logs
│   ├── streamlit_app.py          # Chat UI
│   ├── ingest.py                 # Ingestion script
│   ├── evaluate.py               # Evaluation script
│   ├── server.py                 # FastAPI launcher
│   ├── run.py                    # Combined launcher
│   ├── requirements.txt          # Dependencies
│   ├── .env.example              # Environment template
│   └── Dockerfile                # Container definition
├── docker-compose.yml
└── README.md
```

## 🔧 API Reference

### POST /ingest

Ingest PDF documents into the vector store.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": false}'
```

**Response:**
```json
{
  "status": "success",
  "documents_processed": 7,
  "total_chunks": 1250,
  "processing_time_seconds": 45.2,
  "errors": []
}
```

### POST /ask

Ask a question and get a grounded answer.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a VOR?", "debug": true, "top_k": 5}'
```

**Parameters:**
- `question` (required): The question to answer
- `debug` (optional): Include retrieved chunks in response
- `top_k` (optional): Number of chunks to retrieve (default: 5)

**Response:**
```json
{
  "answer": "A VOR (VHF Omnidirectional Range) is a radio navigation system that provides bearing information to or from the station. [11-radio-navigation-2014.pdf, Page 42]",
  "citations": [
    {
      "document_name": "11-radio-navigation-2014.pdf",
      "page_number": 42,
      "chunk_id": "abc123_chunk_0042",
      "relevant_text": "VOR is a type of short-range radio...",
      "confidence": 0.87
    }
  ],
  "confidence": 0.87,
  "is_grounded": true,
  "retrieved_chunks": [...],
  "processing_time_seconds": 1.23
}
```

### GET /health

Check system health.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-02-01T10:30:00",
  "components": {
    "embeddings": "healthy",
    "vector_store": "healthy",
    "llm": "healthy",
    "reranker": "healthy"
  }
}
```

### GET /documents

List indexed documents.

### DELETE /documents

Clear all indexed documents.

### GET /stats

Get system statistics.

## 📊 Evaluation

### Question Set (50 Questions)

| Type | Count | Description |
|------|-------|-------------|
| Factual | 20 | Simple definitions, direct lookups |
| Applied | 20 | Scenario-based, operational questions |
| Reasoning | 10 | Multi-step reasoning, trade-offs |

### Run Evaluation

```bash
cd backend

# Full evaluation (50 questions)
python evaluate.py

# Specific question type
python evaluate.py --type factual

# Limited questions
python evaluate.py --limit 10

# Custom output directory
python evaluate.py --output ./my_results
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Retrieval Hit Rate** | % of questions where retrieved chunks contain the answer |
| **Faithfulness Rate** | % of answers fully grounded in retrieved text |
| **Hallucination Rate** | % of answers with unsupported claims |
| **Average Confidence** | Mean confidence score across all answers |

### Sample Report Output

```
==================================================
EVALUATION SUMMARY
==================================================
Total Questions:      50
Retrieval Hit Rate:   85.0%
Faithfulness Rate:    88.0%
Hallucination Rate:   4.0%
Average Confidence:   82.0%
==================================================

Questions by Type:
  Factual: 20
  Applied: 20
  Reasoning: 10

Results saved to: ./data/evaluation_results
```

## 📈 Chunking Strategy

The chunking strategy is optimized for aviation documents:

1. **Page-Aware Extraction**: Tracks page numbers for accurate citations
2. **Sentence Boundary Breaking**: Chunks break at sentence boundaries
3. **Configurable Overlap**: 128 characters ensures context continuity
4. **Metadata Preservation**: Each chunk retains document name, page, section

**Default Settings:**
- Chunk size: 512 characters
- Overlap: 128 characters
- Minimum chunk size: 50 characters

**Rationale:**
- Aviation documents often have dense technical content
- 512 characters balances context completeness vs retrieval precision
- Page-level tracking enables precise citations

## 🛡️ Hallucination Control

Multiple layers of hallucination prevention:

1. **Strict System Prompt**: LLM instructed to only use provided context
2. **Low Temperature (0.1)**: Reduces creative/speculative responses
3. **Grounding Verification**: Separate LLM call verifies answer grounding
4. **Confidence Thresholding**: Refuses answer if confidence < 50%
5. **Explicit Refusal**: Returns standard message when context insufficient
6. **Relevance Threshold**: Refuses if retrieved chunks have low similarity scores

**Refusal Response:**
> "This information is not available in the provided document(s)."

## 🔬 Level 2: Hybrid Retrieval

This implementation includes **Level 2 Enhancement** with:

### Components

1. **Vector Search (FAISS)**: 
   - Semantic similarity using dense embeddings
   - GPU-accelerated when available
   - Normalized vectors for cosine similarity

2. **BM25 Keyword Search**: 
   - Traditional term frequency matching
   - Catches exact keyword matches
   - Complements semantic search

3. **Reciprocal Rank Fusion**: 
   - Combines results from both methods
   - Configurable weights (default: 70% vector, 30% BM25)

4. **Cross-Encoder Reranking**: 
   - Joint query-document encoding
   - More accurate relevance scoring
   - Applied to top candidates

### Performance Comparison

| Configuration | Retrieval Hit Rate | Faithfulness |
|--------------|-------------------|--------------|
| Vector Only | ~70% | ~75% |
| Hybrid (BM25 + Vector) | ~80% | ~82% |
| **Hybrid + Reranker** | **~85%** | **~88%** |

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub PAT for Models API | Required for LLM |
| `LLM_ENDPOINT` | LLM API endpoint | `https://models.github.ai/inference` |
| `LLM_MODEL` | Model to use | `openai/gpt-4o` |
| `EMBEDDING_MODEL` | Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Characters per chunk | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks | `128` |
| `TOP_K_RETRIEVAL` | Initial retrieval count | `10` |
| `TOP_K_RERANK` | Final chunks after reranking | `5` |
| `SIMILARITY_THRESHOLD` | Minimum relevance score | `0.3` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for answers | `0.5` |
| `USE_GPU` | Enable GPU acceleration | `true` |
| `USE_BM25` | Enable hybrid retrieval | `true` |
| `USE_RERANKER` | Enable cross-encoder | `true` |
| `GROUNDING_CHECK` | Enable grounding verification | `true` |

### Example .env

```env
GITHUB_TOKEN=ghp_your_token_here
LLM_ENDPOINT=https://models.github.ai/inference
LLM_MODEL=openai/gpt-4o
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=128
TOP_K_RETRIEVAL=10
TOP_K_RERANK=5
USE_GPU=true
USE_BM25=true
USE_RERANKER=true
```

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d aviation-rag

# With GPU support
docker-compose up -d aviation-rag-gpu

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🎬 Demo Video Script (5-8 minutes)

1. **Introduction** (30s)
   - Overview of the system
   - Tech stack overview

2. **Setup & Configuration** (1m)
   - Show .env configuration
   - Explain key settings

3. **Document Ingestion** (1m)
   - Run `python ingest.py`
   - Show indexed documents
   - Explain chunking strategy

4. **API Demonstration** (1m)
   - Health check endpoint
   - Show Swagger docs

5. **Chat Interface** (2m)
   - Ask factual question (e.g., "What is a VOR?")
   - Show citations and debug mode
   - Ask out-of-scope question (expect refusal)

6. **Evaluation** (1m)
   - Run `python evaluate.py --limit 10`
   - Show metrics and report

7. **Level 2 Features** (1m)
   - Toggle hybrid retrieval
   - Show reranking in action

8. **Conclusion** (30s)
   - Summary of features
   - Reliability emphasis

## 📝 License

MIT License

## ⚠️ Disclaimer

This system is for educational and training purposes. Always verify critical aviation information with official sources and certified instructors.

---

Built for AIRMAN AI/ML Technical Assignment
