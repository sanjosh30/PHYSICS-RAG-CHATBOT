"""
src/utils/config.py
-------------------
Centralised configuration using Pydantic Settings.
All values come from environment variables / .env file.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM Backend ---
    llm_backend: str = Field(default="ollama", description="'ollama' or 'openai'")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")

    # --- Embeddings ---
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_device: str = Field(default="cpu")

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(default="./vectorstore/chroma_db")
    chroma_collection_name: str = Field(default="physics_corpus")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=10)
    rerank_top_k: int = Field(default=3)
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # --- Chunking ---
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # --- Generation ---
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=1024)
    out_of_scope_threshold: float = Field(default=0.35)

    # --- Data paths ---
    data_raw_dir: str = Field(default="./data/raw")
    data_processed_dir: str = Field(default="./data/processed")
    benchmark_file: str = Field(default="./data/benchmark/questions.json")

    # --- Logging ---
    log_dir: str = Field(default="./logs")
    log_level: str = Field(default="INFO")

    # --- Derived helpers (not from env) ---
    @property
    def feynman_dir(self) -> Path:
        return Path(self.data_raw_dir) / "feynman"

    @property
    def openstax_dir(self) -> Path:
        return Path(self.data_raw_dir) / "openstax"

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_processed_dir)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir)


# Singleton — import this everywhere
settings = Settings()
