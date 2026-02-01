"""
Aviation RAG Chat - Streamlit User Interface
Modern chat interface for aviation document Q&A
"""

import streamlit as st
import requests
import time
from typing import Dict, Any, Optional
import json

# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Aviation Document AI Chat",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    
    /* Citation styling */
    .citation-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    
    /* Chunk display */
    .chunk-box {
        background-color: #e8f5e9;
        border: 1px solid #4caf50;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    
    /* Status indicators */
    .status-healthy {
        color: #4caf50;
    }
    
    .status-unhealthy {
        color: #f44336;
    }
    
    /* Sidebar styling */
    .sidebar .element-container {
        padding: 0.25rem 0;
    }
    
    /* Grounding badge */
    .grounded-badge {
        background-color: #4caf50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
    }
    
    .ungrounded-badge {
        background-color: #ff9800;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API Functions
# =============================================================================

def check_api_health() -> Dict[str, Any]:
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_stats() -> Dict[str, Any]:
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def ingest_documents(force_reindex: bool = False) -> Dict[str, Any]:
    """Trigger document ingestion."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json={"force_reindex": force_reindex},
            timeout=300,  # 5 minutes for large documents
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def ask_question(question: str, debug: bool = False, top_k: int = 5) -> Dict[str, Any]:
    """Ask a question to the RAG system."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={
                "question": question,
                "debug": debug,
                "top_k": top_k,
            },
            timeout=60,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_documents() -> Dict[str, Any]:
    """Get list of indexed documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Session State Initialization
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "top_k" not in st.session_state:
    st.session_state.top_k = 5


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("✈️ Aviation RAG Chat")
    st.markdown("---")
    
    # System Status
    st.subheader("📊 System Status")
    
    health = check_api_health()
    
    if health.get("status") == "error":
        st.error(f"API not reachable: {health.get('message', 'Unknown error')}")
        st.info("Make sure the FastAPI server is running on http://localhost:8000")
    else:
        status_color = "🟢" if health.get("status") == "healthy" else "🟡"
        st.write(f"{status_color} API Status: **{health.get('status', 'unknown')}**")
        
        if "components" in health:
            with st.expander("Component Details"):
                for comp, status in health["components"].items():
                    icon = "✅" if "healthy" in status else "⚠️"
                    st.write(f"{icon} {comp}: {status}")
    
    # Document Stats
    st.markdown("---")
    st.subheader("📚 Documents")
    
    docs_info = get_documents()
    if "error" not in docs_info:
        st.metric("Total Documents", docs_info.get("total_documents", 0))
        st.metric("Total Chunks", docs_info.get("total_chunks", 0))
        
        with st.expander("Document List"):
            for doc in docs_info.get("documents", []):
                st.write(f"📄 {doc['filename']} ({doc['chunk_count']} chunks)")
    else:
        st.warning("Could not fetch document info")
    
    # Chat Controls
    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# Main Chat Interface
# =============================================================================

st.title("✈️ Aviation Document AI Chat")

st.markdown("""
Ask questions about aviation topics. Answers are generated **only** from the ingested 
aviation documents (PPL/CPL/ATPL textbooks, SOPs, manuals).

If information is not available in the documents, the system will refuse to answer.
""")

st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="✈️"):
            st.markdown(content)
            
            # Show citations if available
            if "citations" in message and message["citations"]:
                with st.expander("📚 Citations"):
                    for citation in message["citations"]:
                        st.markdown(f"""
                        **📄 {citation['document_name']}**, Page {citation['page_number']}
                        
                        > _{citation['relevant_text'][:200]}..._
                        
                        Confidence: {citation['confidence']:.2%}
                        """)
            
            # Show retrieved chunks in debug mode
            if "chunks" in message and message["chunks"]:
                with st.expander("🔍 Retrieved Chunks (Debug)"):
                    for i, chunk in enumerate(message["chunks"], 1):
                        st.markdown(f"""
                        **Chunk {i}** - {chunk['source_file']}, Page {chunk['page_number']}
                        
                        Score: {chunk['score']:.4f}
                        {f"Rerank Score: {chunk.get('rerank_score', 'N/A')}" if chunk.get('rerank_score') else ""}
                        
                        ```
                        {chunk['content'][:500]}...
                        ```
                        """)
            
            # Show metadata
            if "metadata" in message:
                meta = message["metadata"]
                cols = st.columns(4)
                with cols[0]:
                    st.caption(f"⏱️ {meta.get('time', 'N/A')}s")
                with cols[1]:
                    confidence = meta.get('confidence', 0)
                    st.caption(f"📊 {confidence:.0%} confidence")
                with cols[2]:
                    is_grounded = meta.get('is_grounded', False)
                    icon = "✅" if is_grounded else "⚠️"
                    st.caption(f"{icon} {'Grounded' if is_grounded else 'Ungrounded'}")

# Chat input
if prompt := st.chat_input("Ask a question about aviation..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get response from API
    with st.chat_message("assistant", avatar="✈️"):
        with st.spinner("Thinking..."):
            response = ask_question(
                question=prompt,
                debug=st.session_state.debug_mode,
                top_k=st.session_state.top_k,
            )
        
        if "error" in response:
            st.error(f"Error: {response['error']}")
            assistant_message = {
                "role": "assistant",
                "content": f"❌ Error: {response['error']}",
            }
        else:
            answer = response.get("answer", "No answer generated")
            st.markdown(answer)
            
            # Show citations
            citations = response.get("citations", [])
            if citations:
                with st.expander("📚 Citations"):
                    for citation in citations:
                        st.markdown(f"""
                        **📄 {citation['document_name']}**, Page {citation['page_number']}
                        
                        > _{citation['relevant_text'][:200]}..._
                        
                        Confidence: {citation['confidence']:.2%}
                        """)
            
            # Show debug info
            chunks = response.get("retrieved_chunks", [])
            if chunks and st.session_state.debug_mode:
                with st.expander("🔍 Retrieved Chunks (Debug)"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"""
                        **Chunk {i}** - {chunk['source_file']}, Page {chunk['page_number']}
                        
                        Score: {chunk['score']:.4f}
                        {f"Rerank Score: {chunk.get('rerank_score', 'N/A')}" if chunk.get('rerank_score') else ""}
                        
                        ```
                        {chunk['content'][:500]}...
                        ```
                        """)
            
            # Show metadata
            cols = st.columns(4)
            with cols[0]:
                st.caption(f"⏱️ {response.get('processing_time_seconds', 0):.2f}s")
            with cols[1]:
                confidence = response.get('confidence', 0)
                st.caption(f"📊 {confidence:.0%} confidence")
            with cols[2]:
                is_grounded = response.get('is_grounded', False)
                icon = "✅" if is_grounded else "⚠️"
                st.caption(f"{icon} {'Grounded' if is_grounded else 'Ungrounded'}")
            
            # Store message
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "chunks": chunks if st.session_state.debug_mode else None,
                "metadata": {
                    "time": response.get("processing_time_seconds", 0),
                    "confidence": response.get("confidence", 0),
                    "is_grounded": response.get("is_grounded", False),
                }
            }
        
        st.session_state.messages.append(assistant_message)


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.caption("""
**Aviation Document AI Chat** | Powered by RAG with Hybrid Retrieval | 
Answers are strictly grounded in provided aviation documents | 
⚠️ Always verify critical aviation information with official sources
""")
