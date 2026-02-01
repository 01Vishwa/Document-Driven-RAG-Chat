#!/usr/bin/env python
"""
Aviation RAG Chat - Server Runner
Start the FastAPI and Streamlit servers
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

def run_fastapi():
    """Run FastAPI server."""
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

def run_streamlit():
    """Run Streamlit server."""
    # Wait for FastAPI to start
    time.sleep(5)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def main():
    print("=" * 50)
    print("Aviation RAG Chat - Starting Servers")
    print("=" * 50)
    print()
    print("FastAPI:   http://localhost:8000")
    print("API Docs:  http://localhost:8000/docs")
    print("Streamlit: http://localhost:8501")
    print()
    print("=" * 50)
    
    # Start FastAPI in a thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    
    # Start Streamlit in main thread
    run_streamlit()

if __name__ == "__main__":
    main()
