#!/usr/bin/env python
"""
Aviation RAG Chat - API Server Runner
Run this script to start the FastAPI server
"""

import os
import sys
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

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
        "app.api.server:app",
        host=host,
        port=port,
        reload=False,
    )
