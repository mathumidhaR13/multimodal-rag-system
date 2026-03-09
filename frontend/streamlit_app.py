import streamlit as st
import requests
import json
from pathlib import Path

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        color: #e0e0e0;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 1rem 0;
    }
    .chunk-card {
        background: #f8f9fc;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .score-badge {
        background: #667eea;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ok {
        color: #28a745;
        font-weight: 600;
    }
    .status-error {
        color: #dc3545;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────

def check_api_health() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def get_index_stats() -> dict:
    """Fetch vector store stats from the API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/ingest/stats", timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}


def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    """Upload and ingest a PDF via the API."""
    try:
        files = {"file": (filename, file_bytes, "application/pdf")}
        response = requests.post(
            f"{API_BASE_URL}/ingest/",
            files=files,
            timeout=120  # Allow time for large PDFs
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


def query_rag(question: str, top_k: int) -> dict:
    """Send a query to the RAG API."""
    try:
        payload = {"question": question, "top_k": top_k}
        response = requests.post(
            f"{API_BASE_URL}/query/",
            json=payload,
            timeout=120  # Allow time for LLM generation
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


def reset_index() -> bool:
    """Reset the vector store via the API."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/ingest/reset", timeout=10
        )
        return response.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 RAG System")
    st.markdown("---")

    # API Health Check
    st.markdown("### 🔌 API Status")
    api_healthy = check_api_health()

    if api_healthy:
        st.markdown(
            '<p class="status-ok">✅ Backend Connected</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="status-error">❌ Backend Offline</p>',
            unsafe_allow_html=True
        )
        st.warning(
            "Start the FastAPI server:\n\n"
            "`uvicorn app.main:app --reload --port 8000`"
        )

    st.markdown("---")

    # Index Stats
    st.markdown("### 📊 Index Stats")
    stats_data = get_index_stats()

    if stats_data and "index_stats" in stats_data:
        stats = stats_data["index_stats"]
        col1, col2 = st.columns(2)
        col1.metric("Vectors", stats.get("total_vectors", 0))
        col2.metric("Documents", stats.get("total_documents", 0))

        if stats.get("documents"):
            st.markdown("**Ingested Files:**")
            for doc in stats["documents"]:
                st.markdown(f"- 📄 `{doc}`")
    else:
        st.info("No documents ingested yet.")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Query Settings")
    top_k = st.slider(
        "Top K Chunks",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of document chunks to retrieve"
    )

    st.markdown("---")

    # Danger Zone
    st.markdown("### 🗑️ Danger Zone")
    if st.button("Reset Vector Store"):
        if reset_index():
            st.success("✅ Index reset successfully!")
            st.rerun()
        else:
            st.error("❌ Reset failed.")


# ─────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────

st.markdown(
    '<p class="main-header">🧠 Multimodal RAG System</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">'
    'Upload documents → Ask questions → Get AI-powered answers'
    '</p>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])


# ══════════════════════════════════════════
# TAB 1 — Document Upload
# ══════════════════════════════════════════
with tab1:
    st.markdown("### 📤 Upload a PDF Document")
    st.markdown(
        "Upload any PDF document to ingest it into the RAG system. "
        "The document will be parsed, chunked, embedded and stored "
        "in the vector database."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Maximum file size: 50MB"
    )

    if uploaded_file is not None:
        # File info preview
        file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 2)
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Filename", uploaded_file.name)
        col2.metric("📦 Size", f"{file_size_kb} KB")
        col3.metric("📋 Type", "PDF")

        st.markdown("---")

        if st.button("🚀 Ingest Document"):
            if not api_healthy:
                st.error(
                    "❌ Cannot ingest — FastAPI backend is offline!"
                )
            else:
                with st.spinner(
                    "⏳ Ingesting document... "
                    "This may take a moment."
                ):
                    result, status_code = ingest_pdf(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name
                    )

                if status_code == 200:
                    st.success("🎉 Document ingested successfully!")

                    # Show results
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "📦 Chunks Created",
                        result.get("total_chunks", 0)
                    )
                    col2.metric(
                        "📄 Filename",
                        result.get("filename", "")
                    )
                    col3.metric(
                        "✅ Status",
                        result.get("status", "").upper()
                    )

                    st.info(
                        "💡 Now go to the **Ask Questions** tab "
                        "to query your document!"
                    )

                else:
                    st.error(
                        f"❌ Ingestion failed: "
                        f"{result.get('detail', 'Unknown error')}"
                    )


# ══════════════════════════════════════════
# TAB 2 — Query Interface
# ══════════════════════════════════════════
with tab2:
    st.markdown("### 💬 Ask a Question")
    st.markdown(
        "Type a question about your uploaded documents. "
        "The system will retrieve relevant chunks and generate an answer."
    )

    # Question input
    question = st.text_area(
        "Your Question",
        placeholder="e.g. What is the main topic of this document?",
        height=100
    )

    # Example questions
    with st.expander("💡 Example Questions"):
        examples = [
            "What is the main topic of this document?",
            "Summarize the key points.",
            "What conclusions are drawn?",
            "What are the main findings?",
            "Who are the key people mentioned?"
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                question = ex

    if st.button("🔍 Search & Answer"):
        if not api_healthy:
            st.error("❌ Cannot query — FastAPI backend is offline!")
        elif not question.strip():
            st.warning("⚠️ Please enter a question first.")
        else:
            with st.spinner(
                "🧠 Thinking... Retrieving chunks and generating answer..."
            ):
                result, status_code = query_rag(
                    question=question,
                    top_k=top_k
                )

            if status_code == 200:

                # ── Answer Box ──
                st.markdown("### 💡 Answer")
                st.markdown(
                    f'<div class="answer-box">'
                    f'{result.get("answer", "No answer generated.")}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # ── Metadata ──
                col1, col2 = st.columns(2)
                col1.metric(
                    "📦 Chunks Retrieved",
                    result.get("total_chunks_retrieved", 0)
                )
                col2.metric(
                    "❓ Question Length",
                    f"{len(question)} chars"
                )

                # ── Retrieved Chunks ──
                st.markdown("---")
                st.markdown("### 📚 Retrieved Context Chunks")
                st.caption(
                    "These are the document sections used "
                    "to generate the answer."
                )

                chunks = result.get("retrieved_chunks", [])
                for i, chunk in enumerate(chunks):
                    with st.expander(
                        f"📄 Chunk {i+1} — "
                        f"Score: {chunk.get('score', 0):.4f} | "
                        f"Source: {chunk.get('source', 'Unknown')}"
                    ):
                        st.markdown(
                            f'<span class="score-badge">'
                            f'Score: {chunk.get("score", 0):.4f}'
                            f'</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"**Source:** `{chunk.get('source', 'N/A')}`"
                        )
                        st.markdown("**Text:**")
                        st.markdown(
                            f"> {chunk.get('text', '')[:500]}..."
                        )

            elif status_code == 400:
                st.warning(
                    f"⚠️ {result.get('detail', 'Bad request')}"
                )
            else:
                st.error(
                    f"❌ Query failed: "
                    f"{result.get('detail', 'Unknown error')}"
                )

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built with FastAPI · FAISS · "
    "Sentence Transformers · FLAN-T5 · Streamlit</small></center>",
    unsafe_allow_html=True
)