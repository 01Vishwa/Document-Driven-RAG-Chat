"""
Aviation RAG Chat - Streamlit Frontend
Interactive chat interface for aviation document Q&A
"""

import streamlit as st
import requests
import json
import os
import glob
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
    .citation-box {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        font-size: 0.9rem;
        border-left: 4px solid #4CAF50;
        color: #333333 !important;
    }
    .citation-box * {
        color: #333333 !important;
    }
    .citation-box small {
        color: #666666 !important;
    }
    .confidence-high { color: #28a745 !important; font-weight: bold; }
    .confidence-medium { color: #ffc107 !important; font-weight: bold; }
    .confidence-low { color: #dc3545 !important; font-weight: bold; }
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
    
    # Navigation
    page = st.radio("Navigation", ["Chat", "Evaluation Results"])
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
    debug_mode = st.toggle("Debug Mode", value=False, help="Show retrieved chunks and routing info")
    
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


if page == "Chat":
    # Main Chat Area
    st.title("✈️ Aviation Document AI Chat")

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
                        doc_name = citation.get("document_name", "Unknown")
                        page_num = citation.get("page_number", "N/A")
                        relevant_text = citation.get("relevant_text", "")[:200]
                        st.info(f"📄 **{doc_name}** - Page {page_num}\n\n_{relevant_text}..._")
            
            # Show routing info in debug mode (only if present in history)
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
            
            # Get response - Default settings: debug=False, top_k=5
            with st.chat_message("assistant"):
                with st.spinner("Searching aviation documents..."):
                    # Hardcoded settings as UI controls were removed
                    response = ask_question(prompt, debug=debug_mode, top_k=5)
                
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
                            doc_name = citation.get("document_name", "Unknown")
                            page_num = citation.get("page_number", "N/A")
                            relevant_text = citation.get("relevant_text", "")[:200]
                            st.info(f"📄 **{doc_name}** - Page {page_num}\n\n_{relevant_text}..._")
                    
                    # Debug info is hidden by default now as controls are removed
                    if debug_mode:
                        
                        # Routing Info
                        if routing_info:
                            with st.expander("🔍 Debug: Routing Info", expanded=True):
                                st.json(routing_info)
                        
                        # Retrieved Chunks
                        chunks = response.get("retrieved_chunks", [])
                        if chunks:
                            with st.expander(f"🔍 Debug: Retrieved Chunks ({len(chunks)})", expanded=True):
                                for i, chunk in enumerate(chunks):
                                    st.markdown(f"**Chunk {i+1}** (Score: {chunk.get('score', 0):.4f})")
                                    st.caption(f"Source: {chunk.get('metadata', {}).get('filename', 'Unknown')} - Page {chunk.get('metadata', {}).get('page_number', '?')}")
                                    st.text(chunk.get('content', '')[:300] + "...")
                                    st.divider()
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "routing_info": routing_info
                })

    # Footer
    st.markdown("---")

elif page == "Evaluation Results":
    st.title("📊 Evaluation Results")
    st.caption("Performance metrics on company-provided question sets")
    
    # Find evaluation reports
    report_dir = "data/company_results"
    
    if os.path.exists(report_dir):
        # Look for markdown reports
        report_files = glob.glob(os.path.join(report_dir, "report_*.md"))
        
        if report_files:
            # Sort by modification time (newest first)
            report_files.sort(key=os.path.getmtime, reverse=True)
            latest_report = report_files[0]
            
            st.success(f"Displaying latest report: `{os.path.basename(latest_report)}`")
            
            with open(latest_report, "r", encoding="utf-8") as f:
                report_content = f.read()
            
            st.markdown(report_content)
            
            # Additional Option: visual exploration of the raw JSON used for this report?
            # We can find the matching JSON file
            json_file = latest_report.replace("report_", "evaluation_results_").replace(".md", ".json")
            if os.path.exists(json_file):
                 with st.expander("📂 View Raw Results Data"):
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    st.json(data)
        else:
            st.warning("No evaluation reports found.")
    else:
        st.warning(f"Directory not found: {report_dir}")
        st.info("Run the evaluation script first to generate results.")
