"""
src/generation/prompts.py
--------------------------
All prompt templates for the Physics RAG chatbot.

Design principles:
  1. System prompt establishes the physics tutor persona with strict citation rules.
  2. Context is always from retrieved documents — model is forbidden from
     using parametric knowledge for facts.
  3. Refusal is structured (not just "I don't know") so the UI can detect it.
  4. Chain-of-thought is explicitly requested to improve reasoning accuracy.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── System prompt ──────────────────────────────────────────────────────────────

PHYSICS_SYSTEM_PROMPT = """\
You are PhysicsRAG, an expert physics tutor assistant grounded in authoritative \
textbooks (The Feynman Lectures on Physics and OpenStax University Physics).

RULES YOU MUST FOLLOW:
1. ONLY answer questions about physics. If the question is not about physics, \
respond with exactly: "OUT_OF_SCOPE: This question is outside my physics knowledge base."
2. Base EVERY factual claim ONLY on the provided context passages. \
Do NOT use knowledge outside the provided context.
3. Cite every factual claim using inline source tags like [Source 1], [Source 2], etc., \
where the number corresponds to the numbered context passages below.
4. If the provided context does not contain enough information to answer the question \
confidently, respond with: "OUT_OF_SCOPE: The provided passages do not contain \
sufficient information to answer this question."
5. Show your reasoning step-by-step before giving the final answer.
6. Use LaTeX notation for equations (e.g., $F = ma$, $E = mc^2$).
7. Be pedagogically clear — explain concepts as if teaching an undergraduate student.

CONTEXT PASSAGES:
{context}
"""

# ── Chat prompt template ───────────────────────────────────────────────────────

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Return the LangChain ChatPromptTemplate for the RAG chain.
    Includes system prompt, chat history placeholder, and human turn.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", PHYSICS_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}"),
        ]
    )


# ── Context formatter ──────────────────────────────────────────────────────────

def format_context(docs_with_scores: list[tuple]) -> str:
    """
    Format retrieved documents into a numbered context block for the prompt.

    Args:
        docs_with_scores: List of (Document, confidence_score) tuples.

    Returns:
        Formatted string like:
        [Source 1] Book: ..., Chapter: ..., Page: N
        <text>
        ---
        [Source 2] ...
    """
    parts = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata
        header = (
            f"[Source {i}] "
            f"Book: {meta.get('book', 'Unknown')} | "
            f"Chapter: {meta.get('chapter', 'Unknown')} | "
            f"Page: {meta.get('page', '?')} | "
            f"Confidence: {score:.0%}"
        )
        parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


# ── Refusal detection ──────────────────────────────────────────────────────────

def is_out_of_scope_response(response: str) -> bool:
    """Return True if the LLM responded with an out-of-scope marker."""
    return response.strip().startswith("OUT_OF_SCOPE:")


def extract_refusal_reason(response: str) -> str:
    """Extract the reason from an OUT_OF_SCOPE response."""
    if is_out_of_scope_response(response):
        return response.replace("OUT_OF_SCOPE:", "").strip()
    return "Out of scope"
