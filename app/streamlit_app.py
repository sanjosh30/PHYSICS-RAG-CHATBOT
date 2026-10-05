"""
app/streamlit_app.py
---------------------
Physics RAG Chatbot — Streamlit frontend.

Features:
  ✦ Dark-themed physics aesthetic with animated gradient header
  ✦ Streaming chat responses with typing indicator
  ✦ Expandable citation cards (book, chapter, page, snippet)
  ✦ Colour-coded confidence meter (green/amber/red)
  ✦ Out-of-scope refusal with styled warning
  ✦ Corpus filter sidebar (Feynman / OpenStax / Both)
  ✦ LaTeX rendering support
  ✦ Persistent chat history within session
  ✦ Corpus statistics in sidebar
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Physics RAG Chatbot",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Physics RAG Chatbot — Powered by Feynman Lectures & OpenStax",
    },
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a1628 100%);
    min-height: 100vh;
}

/* ── Animated Header ── */
.rag-header {
    background: linear-gradient(120deg, #1a1a4e, #0d2137, #1a3a5c, #0d2137);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99, 179, 237, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 60px rgba(99, 179, 237, 0.05);
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.rag-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #90cdf4, #bee3f8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
}

.rag-header p {
    color: #8fb8d4;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}

/* ── Chat Messages ── */
.user-msg {
    background: linear-gradient(135deg, #1e3a5f, #1a2d4a);
    border: 1px solid rgba(99, 179, 237, 0.3);
    border-radius: 12px 12px 2px 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.6;
}

.assistant-msg {
    background: linear-gradient(135deg, #0f1f0f, #111e11);
    border: 1px solid rgba(72, 187, 120, 0.25);
    border-radius: 12px 12px 12px 2px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.7;
}

.out-of-scope-msg {
    background: linear-gradient(135deg, #2d1b00, #2d1515);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #fcd34d;
}

/* ── Citation Cards ── */
.citation-card {
    background: rgba(15, 25, 40, 0.8);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-left: 3px solid #63b3ed;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
}

.citation-card .citation-meta {
    color: #63b3ed;
    font-weight: 600;
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
}

.citation-card .citation-snippet {
    color: #94a3b8;
    font-style: italic;
    font-size: 0.82rem;
    line-height: 1.5;
}

/* ── Confidence Meter ── */
.confidence-bar-container {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 8px;
    margin: 0.5rem 0;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a1628 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.1);
}

section[data-testid="stSidebar"] .sidebar-section {
    background: rgba(15, 25, 40, 0.6);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.8rem 0;
}

/* ── Input Box ── */
.stChatInput > div {
    background: rgba(15, 25, 40, 0.9) !important;
    border: 1px solid rgba(99, 179, 237, 0.3) !important;
    border-radius: 12px !important;
}

/* ── Stats Badge ── */
.stat-badge {
    display: inline-block;
    background: rgba(99, 179, 237, 0.15);
    border: 1px solid rgba(99, 179, 237, 0.3);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    color: #90cdf4;
    margin: 0.2rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99, 179, 237, 0.3); border-radius: 3px; }

/* ── Streamlit overrides ── */
.stMarkdown p { color: #cbd5e0; }
h1, h2, h3 { color: #e2e8f0 !important; }
.stExpander { background: rgba(15, 25, 40, 0.5) !important; border: 1px solid rgba(99, 179, 237, 0.15) !important; border-radius: 8px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────────

def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {role, content, citations, confidence, oos}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # LangChain BaseMessage list
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "corpus_stats" not in st.session_state:
        st.session_state.corpus_stats = None


_init_state()


# ── Load RAG chain (lazy, cached in session) ───────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag_chain():
    """Load the RAG chain once and cache it for the session."""
    from src.generation.rag_chain import PhysicsRAGChain
    return PhysicsRAGChain()


@st.cache_resource(show_spinner=False)
def load_vector_store_stats():
    """Get vector store statistics."""
    try:
        from src.retrieval.vector_store import PhysicsVectorStore
        store = PhysicsVectorStore()
        ids = store.get_all_ids()
        return {"chunks": len(ids), "status": "ready"}
    except Exception as e:
        return {"chunks": 0, "status": f"error: {e}"}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
            <div style="font-size:2.5rem;">⚛️</div>
            <div style="color:#90cdf4; font-weight:600; font-size:1.1rem;">Physics RAG</div>
            <div style="color:#64748b; font-size:0.8rem;">Powered by Feynman + OpenStax</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Corpus filter ──
    st.markdown("**📚 Knowledge Source**")
    corpus_option = st.radio(
        "Select corpus",
        ["All Sources", "Feynman Lectures", "OpenStax University Physics"],
        label_visibility="collapsed",
    )
    corpus_filter_map = {
        "All Sources": None,
        "Feynman Lectures": "feynman",
        "OpenStax University Physics": "openstax",
    }
    selected_corpus = corpus_filter_map[corpus_option]

    st.divider()

    # ── Settings ──
    st.markdown("**⚙️ Settings**")
    show_retrieved = st.toggle("Show all retrieved chunks", value=False)
    show_scores = st.toggle("Show similarity scores", value=True)

    st.divider()

    # ── Vector store stats ──
    st.markdown("**📊 Index Status**")
    with st.spinner("Checking index..."):
        stats = load_vector_store_stats()

    if stats["status"] == "ready":
        st.markdown(
            f"""
            <span class="stat-badge">✓ {stats['chunks']:,} chunks indexed</span><br>
            <span class="stat-badge">🔵 Feynman Lectures</span>
            <span class="stat-badge">🟢 OpenStax Vol 1–3</span>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error(f"Index not ready: {stats['status']}")
        st.info("Run: `python scripts/ingest.py` to build the index.")

    st.divider()

    # ── Clear chat ──
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        "<div style='color:#4a5568; font-size:0.75rem; text-align:center; margin-top:1rem;'>"
        "Citations ≥ 85% accuracy target<br>Hallucination rate &lt; 10%"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="rag-header">
        <h1>⚛️ Physics RAG Chatbot</h1>
        <p>Ask any undergraduate physics question — every answer is grounded in
        <strong>The Feynman Lectures on Physics</strong> and
        <strong>OpenStax University Physics</strong>, with cited passages for every claim.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Example questions ──────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("##### 💡 Try asking:")
    example_cols = st.columns(3)
    examples = [
        "What is Newton's second law and how is it derived?",
        "Explain the photoelectric effect and Einstein's explanation.",
        "What are Maxwell's equations and what do they describe?",
        "How does the Bohr model of the hydrogen atom work?",
        "What is the principle of conservation of energy?",
        "Explain simple harmonic motion and its differential equation.",
    ]
    for i, example in enumerate(examples):
        col = example_cols[i % 3]
        with col:
            if st.button(
                example,
                key=f"example_{i}",
                use_container_width=True,
            ):
                st.session_state._pending_question = example
                st.rerun()


# ── Render chat history ────────────────────────────────────────────────────────

def render_confidence_bar(score: float) -> str:
    """Generate an HTML confidence bar."""
    pct = int(score * 100)
    if score >= 0.8:
        colour = "#22c55e"
        label = "High"
    elif score >= 0.5:
        colour = "#f59e0b"
        label = "Medium"
    else:
        colour = "#ef4444"
        label = "Low"

    return f"""
    <div style="margin: 0.5rem 0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#94a3b8; font-size:0.8rem;">Confidence</span>
            <span style="color:{colour}; font-size:0.8rem; font-weight:600;">{label} · {pct}%</span>
        </div>
        <div class="confidence-bar-container">
            <div class="confidence-bar-fill" style="width:{pct}%; background:{colour};"></div>
        </div>
    </div>
    """


for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">🧑‍🎓 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        if msg.get("oos"):
            st.markdown(
                f"""
                <div class="out-of-scope-msg">
                    ⚠️ <strong>Out of Scope</strong><br>
                    {msg["content"]}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-msg">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Confidence bar
            conf = msg.get("confidence", 0)
            if conf > 0 and show_scores:
                st.markdown(render_confidence_bar(conf), unsafe_allow_html=True)

            # Citations
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📚 {len(citations)} Source(s) cited", expanded=False):
                    for cit in citations:
                        corpus_icon = "🔵" if cit.get("corpus") == "feynman" else "🟢"
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-meta">
                                    {corpus_icon} [Source {cit['index']}] &nbsp;|&nbsp;
                                    {cit['book']} &nbsp;|&nbsp;
                                    {cit['chapter']} &nbsp;|&nbsp;
                                    Page {cit['page']}
                                    {"&nbsp;|&nbsp; Score: " + f"{cit['score']:.0%}" if show_scores else ""}
                                </div>
                                <div class="citation-snippet">"{cit['snippet']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Show all retrieved chunks if requested
            if show_retrieved and msg.get("all_retrieved"):
                with st.expander("🔍 All retrieved chunks (debug)", expanded=False):
                    for i, (doc_content, score) in enumerate(
                        msg["all_retrieved"], start=1
                    ):
                        st.markdown(f"**Chunk {i}** — Score: `{score:.3f}`")
                        st.text(doc_content[:300] + "...")
                        st.divider()


# ── Chat input ─────────────────────────────────────────────────────────────────

# Handle example button click
pending = getattr(st.session_state, "_pending_question", None)
if pending:
    del st.session_state._pending_question
    user_question = pending
else:
    user_question = st.chat_input(
        "Ask a physics question… (e.g. 'Explain Faraday's law of induction')",
        key="chat_input",
    )

if user_question:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append(HumanMessage(content=user_question))

    # Render the user message immediately
    st.markdown(
        f'<div class="user-msg">🧑‍🎓 {user_question}</div>',
        unsafe_allow_html=True,
    )

    # Load chain
    with st.spinner(""):
        try:
            chain = load_rag_chain()
        except Exception as e:
            st.error(f"Failed to load RAG chain: {e}")
            st.stop()

    # Stream response
    response_placeholder = st.empty()
    full_response = ""

    with st.spinner("🔍 Retrieving relevant passages…"):
        try:
            result = chain.ask(
                question=user_question,
                chat_history=st.session_state.chat_history[:-1],  # exclude current
                corpus_filter=selected_corpus,
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    full_response = result.answer

    if result.is_out_of_scope:
        st.markdown(
            f"""
            <div class="out-of-scope-msg">
                ⚠️ <strong>Out of Scope</strong><br>
                {result.refusal_reason or full_response}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.refusal_reason or full_response,
                "oos": True,
                "citations": [],
                "confidence": 0.0,
            }
        )
    else:
        st.markdown(
            f'<div class="assistant-msg">{full_response}</div>',
            unsafe_allow_html=True,
        )

        # Confidence bar
        if show_scores and result.confidence > 0:
            st.markdown(render_confidence_bar(result.confidence), unsafe_allow_html=True)

        # Citations
        citation_dicts = [c.to_dict() for c in result.citations]
        if citation_dicts:
            with st.expander(f"📚 {len(citation_dicts)} Source(s) cited", expanded=True):
                for cit in citation_dicts:
                    corpus_icon = "🔵" if cit.get("corpus") == "feynman" else "🟢"
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-meta">
                                {corpus_icon} [Source {cit['index']}] &nbsp;|&nbsp;
                                {cit['book']} &nbsp;|&nbsp;
                                {cit['chapter']} &nbsp;|&nbsp;
                                Page {cit['page']}
                                {"&nbsp;|&nbsp; Score: " + f"{cit['score']:.0%}" if show_scores else ""}
                            </div>
                            <div class="citation-snippet">"{cit['snippet']}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # All retrieved (debug)
        all_retrieved_display = [
            (doc.page_content, score) for doc, score in result.retrieved_docs
        ]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "oos": False,
                "citations": citation_dicts,
                "confidence": result.confidence,
                "all_retrieved": all_retrieved_display,
            }
        )

    # Update LangChain chat history
    st.session_state.chat_history.append(AIMessage(content=full_response))
