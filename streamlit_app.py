"""
Aviation RAG Chat - Streamlit Frontend
Interactive chat interface for aviation document Q&A
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Aviation Document AI Chat",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChat message {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .citation-box {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        font-size: 0.85rem;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .routing-info {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None


def get_stats():
    """Get system statistics."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        return response.json()
    except:
        return None


def get_documents():
    """Get list of indexed documents."""
    try:
        response = requests.get(f"{API_URL}/documents", timeout=5)
        return response.json()
    except:
        return None


def ingest_documents():
    """Trigger document ingestion."""
    try:
        response = requests.post(f"{API_URL}/ingest", timeout=300)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def ask_question(question: str, debug: bool = False, top_k: int = 5):
    """Send question to the API."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "debug": debug, "top_k": top_k},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# Sidebar
with st.sidebar:
    st.title("✈️ Aviation AI Chat")
    st.markdown("---")
    
    # API Status
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("🟢 API Connected")
        if health_data:
            with st.expander("Component Status"):
                for component, status in health_data.get("components", {}).items():
                    icon = "✅" if "healthy" in status else "⚠️"
                    st.text(f"{icon} {component}: {status}")
    else:
        st.error("🔴 API Disconnected")
        st.info("Start the API server:\n```\npython scripts/run_api.py\n```")
    
    st.markdown("---")
    
    # Document Stats
    st.subheader("📚 Indexed Documents")
    docs = get_documents()
    if docs:
        st.metric("Total Documents", docs.get("total_documents", 0))
        st.metric("Total Chunks", docs.get("total_chunks", 0))
        
        with st.expander("Document List"):
            for doc in docs.get("documents", []):
                st.text(f"📄 {doc['filename']}")
                st.caption(f"   {doc['chunk_count']} chunks")
    else:
        st.warning("No documents indexed")
        if st.button("🔄 Ingest Documents"):
            with st.spinner("Ingesting documents..."):
                result = ingest_documents()
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"Ingested {result.get('documents_processed', 0)} documents!")
                    st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    debug_mode = st.checkbox("Debug Mode", value=False, help="Show retrieved chunks and routing info")
    top_k = st.slider("Retrieval Top-K", min_value=3, max_value=20, value=5)
    
    st.markdown("---")
    
    # System Info
    stats = get_stats()
    if stats:
        with st.expander("System Info"):
            st.json({
                "embedding_model": stats.get("embedding_model", "N/A"),
                "llm_model": stats.get("llm_model", "N/A"),
                "reranker_enabled": stats.get("reranker_enabled", False),
                "bm25_enabled": stats.get("bm25_enabled", False),
                "query_router_enabled": stats.get("query_router_enabled", False),
            })


# Main Chat Area
st.title("✈️ Aviation Document AI Chat")
st.caption("Ask questions about aviation documents - PPL/CPL/ATPL textbooks, SOPs, and manuals")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show citations for assistant messages
        if message["role"] == "assistant" and "citations" in message:
            if message["citations"]:
                st.markdown("**📎 Citations:**")
                for citation in message["citations"]:
                    st.markdown(
                        f'<div class="citation-box">'
                        f'📄 **{citation["document_name"]}** - Page {citation["page_number"]}<br>'
                        f'<small>{citation["relevant_text"][:150]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        # Show routing info in debug mode
        if message["role"] == "assistant" and "routing_info" in message and message["routing_info"]:
            with st.expander("🔍 Debug: Routing Info"):
                st.json(message["routing_info"])


# Chat input
if prompt := st.chat_input("Ask a question about aviation..."):
    # Check API
    if not is_healthy:
        st.error("API is not connected. Please start the server first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching aviation documents..."):
                response = ask_question(prompt, debug=debug_mode, top_k=top_k)
            
            if "error" in response:
                st.error(f"Error: {response['error']}")
                answer = "I encountered an error processing your question."
                citations = []
                routing_info = None
            else:
                answer = response.get("answer", "No answer received.")
                citations = response.get("citations", [])
                confidence = response.get("confidence", 0)
                is_grounded = response.get("is_grounded", False)
                routing_info = response.get("routing_info", None)
                
                # Display answer
                st.markdown(answer)
                
                # Confidence indicator
                if confidence >= 0.8:
                    conf_class = "confidence-high"
                    conf_icon = "🟢"
                elif confidence >= 0.5:
                    conf_class = "confidence-medium"
                    conf_icon = "🟡"
                else:
                    conf_class = "confidence-low"
                    conf_icon = "🔴"
                
                st.markdown(
                    f'{conf_icon} **Confidence:** <span class="{conf_class}">{confidence:.0%}</span> | '
                    f'**Grounded:** {"✅" if is_grounded else "⚠️"}',
                    unsafe_allow_html=True
                )
                
                # Citations
                if citations:
                    st.markdown("**📎 Citations:**")
                    for citation in citations:
                        st.markdown(
                            f'<div class="citation-box">'
                            f'📄 **{citation["document_name"]}** - Page {citation["page_number"]}<br>'
                            f'<small>{citation.get("relevant_text", "")[:150]}...</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                
                # Debug: Retrieved chunks
                if debug_mode and response.get("retrieved_chunks"):
                    with st.expander(f"🔍 Debug: Retrieved Chunks ({len(response['retrieved_chunks'])})"):
                        for i, chunk in enumerate(response["retrieved_chunks"], 1):
                            st.markdown(f"**Chunk {i}** (Score: {chunk['score']:.3f})")
                            st.text(f"Source: {chunk['source_file']}, Page {chunk['page_number']}")
                            st.text(chunk["content"][:300] + "...")
                            st.markdown("---")
                
                # Debug: Routing info
                if debug_mode and routing_info:
                    with st.expander("🔍 Debug: Query Routing"):
                        st.json(routing_info)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "routing_info": routing_info if debug_mode else None
            })


# Footer
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer**: This system answers ONLY from provided aviation documents. "
    "If information is not available, it will explicitly refuse. "
    "Always verify critical information with official sources."
)
