import subprocess
import sys
import time
import os

def main():
    print("Starting Aviation RAG Chat...")
    print("--------------------------------")
    
    env = os.environ.copy()
    
    # Start Backend
    print("launching API server...")
    backend_process = subprocess.Popen(
        [sys.executable, "scripts/run_api.py"],
        env=env
    )
    
    # Wait for backend to be ready (naive wait)
    time.sleep(3)
    
    # Start Frontend
    print("launching Streamlit UI...")
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"],
        env=env
    )
    
    print("--------------------------------")
    print("Services running. Press Ctrl+C to stop.")
    print("API: http://localhost:8000")
    print("UI:  http://localhost:8501")
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print("Stopped.")

if __name__ == "__main__":
    main()
