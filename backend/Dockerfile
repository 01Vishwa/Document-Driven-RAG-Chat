# Dockerfile for Aviation RAG Chat
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install faiss-gpu if CUDA available (fallback to cpu)
RUN pip install --no-cache-dir faiss-cpu || true

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/faiss_index /app/data/processed /app/logs

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "run.py"]
