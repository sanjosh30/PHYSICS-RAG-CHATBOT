"""
src/generation/rag_chain.py
----------------------------
The main RAG chain: ties together retrieval, reranking, prompt building,
LLM generation, and citation extraction.

Supports:
  - Ollama (Llama 3.1) for local inference
  - OpenAI GPT-4o-mini as a fallback
  - Streaming responses for Streamlit
  - Chat history for multi-turn conversations

Returns a structured RAGResult with answer, citations, and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.generation.citation_engine import Citation, CitationEngine
from src.generation.prompts import (
    extract_refusal_reason,
    format_context,
    get_rag_prompt,
    is_out_of_scope_response,
)
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import PhysicsRetriever
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResult:
    """Structured result from the RAG chain."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    is_out_of_scope: bool = False
    refusal_reason: str = ""
    retrieved_docs: list[tuple[Document, float]] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0


def _build_llm():
    """Instantiate the configured LLM."""
    if settings.llm_backend == "ollama":
        logger.info(
            f"Using Ollama: {settings.ollama_model} @ {settings.ollama_base_url}"
        )
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )
    else:
        logger.info(f"Using OpenAI: {settings.openai_model}")
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )


class PhysicsRAGChain:
    """
    Production RAG chain for physics Q&A.

    Usage:
        chain = PhysicsRAGChain()
        result = chain.ask("What is the speed of light?")
        print(result.answer)
        for citation in result.citations:
            print(citation.short_ref)
    """

    def __init__(self) -> None:
        self._retriever = PhysicsRetriever()
        self._reranker = Reranker()
        self._citation_engine = CitationEngine()
        self._llm = _build_llm()
        self._prompt = get_rag_prompt()
        self._chain = self._prompt | self._llm | StrOutputParser()

        logger.info("PhysicsRAGChain initialised")

    def ask(
        self,
        question: str,
        chat_history: list[BaseMessage] | None = None,
        corpus_filter: str | None = None,
    ) -> RAGResult:
        """
        Answer a physics question with retrieved context and citations.

        Args:
            question:      The user's question.
            chat_history:  Previous conversation messages for multi-turn support.
            corpus_filter: Optionally restrict to "feynman" or "openstax".

        Returns:
            RAGResult with answer, citations, and confidence.
        """
        logger.info(f"Processing question: '{question[:80]}'")

        # 1. Retrieve
        retrieved = self._retriever.retrieve(
            question,
            k=settings.retrieval_top_k,
            corpus_filter=corpus_filter,
        )

        # 2. Rerank → check if in scope
        reranked = self._reranker.rerank(
            question, retrieved, top_n=settings.rerank_top_k
        )

        if not self._reranker.is_in_scope(reranked):
            logger.info("Query deemed out-of-scope by reranker threshold")
            return RAGResult(
                answer="OUT_OF_SCOPE: The provided passages do not contain "
                       "sufficient information to answer this question.",
                is_out_of_scope=True,
                refusal_reason="No sufficiently relevant passages found in the physics corpus.",
                retrieved_docs=reranked,
            )

        # 3. Build context
        context_str = format_context(reranked)

        # 4. Generate
        logger.info("Generating answer with LLM...")
        chain_input = {
            "context": context_str,
            "question": question,
            "chat_history": chat_history or [],
        }

        raw_answer: str = self._chain.invoke(chain_input)

        # 5. Check for LLM-generated refusal
        if is_out_of_scope_response(raw_answer):
            reason = extract_refusal_reason(raw_answer)
            return RAGResult(
                answer=raw_answer,
                is_out_of_scope=True,
                refusal_reason=reason,
                retrieved_docs=reranked,
            )

        # 6. Extract citations
        citations = self._citation_engine.extract(raw_answer, reranked)
        confidence = self._citation_engine.aggregate_confidence(reranked)

        logger.info(
            f"Answer generated. Citations: {len(citations)}, "
            f"Confidence: {confidence:.2f}"
        )

        return RAGResult(
            answer=raw_answer,
            citations=citations,
            confidence=confidence,
            retrieved_docs=reranked,
        )

    def stream(
        self,
        question: str,
        chat_history: list[BaseMessage] | None = None,
        corpus_filter: str | None = None,
    ) -> Generator[str, None, RAGResult]:
        """
        Stream the LLM response token by token, then return the full RAGResult.

        Usage in Streamlit:
            result = None
            for token in chain.stream(question):
                st.write(token)
        Note: yields str tokens; use the return value for citations.
        """
        # Retrieve & rerank (same as ask())
        retrieved = self._retriever.retrieve(
            question, k=settings.retrieval_top_k, corpus_filter=corpus_filter
        )
        reranked = self._reranker.rerank(
            question, retrieved, top_n=settings.rerank_top_k
        )

        if not self._reranker.is_in_scope(reranked):
            refusal = (
                "⚠️ This question falls outside my physics knowledge base. "
                "Please ask about topics covered in the Feynman Lectures or "
                "OpenStax University Physics."
            )
            yield refusal
            return RAGResult(
                answer=refusal,
                is_out_of_scope=True,
                refusal_reason="Reranker confidence below threshold.",
                retrieved_docs=reranked,
            )

        context_str = format_context(reranked)
        chain_input = {
            "context": context_str,
            "question": question,
            "chat_history": chat_history or [],
        }

        full_response = ""
        for chunk in self._chain.stream(chain_input):
            full_response += chunk
            yield chunk

        citations = self._citation_engine.extract(full_response, reranked)
        confidence = self._citation_engine.aggregate_confidence(reranked)

        return RAGResult(
            answer=full_response,
            citations=citations,
            confidence=confidence,
            retrieved_docs=reranked,
            is_out_of_scope=is_out_of_scope_response(full_response),
        )
