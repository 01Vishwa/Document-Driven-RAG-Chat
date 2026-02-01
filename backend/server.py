#!/usr/bin/env python
"""
Aviation RAG Chat - Server Launcher
Run this script to start the FastAPI server
"""

import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    import uvicorn
    
    # Load config without triggering all imports
    from dotenv import load_dotenv
    load_dotenv()
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print("=" * 60)
    print("Aviation RAG Chat - Starting Server")
    print("=" * 60)
    print(f"API: http://localhost:{port}")
    print(f"Docs: http://localhost:{port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
    )
