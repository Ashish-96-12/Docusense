"""App settings, loaded from environment variables or a .env file.

Every setting can be overridden with an env var prefixed DOCUSENSE_,
e.g. DOCUSENSE_LLM_PROVIDER=openai. API keys use their standard names.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["auto", "anthropic", "openai", "ollama", "extractive"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCUSENSE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Storage
    data_dir: Path = Path("data/store")
    max_upload_mb: int = 25

    # Chunking (sizes are in words)
    chunk_size: int = 220
    chunk_overlap: int = 40

    # Embeddings: "auto" tries sentence-transformers and falls back to a
    # hashing embedder if the model can't be loaded (e.g. no internet).
    embedding_backend: Literal["auto", "sentence-transformers", "hashing"] = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    retrieval_mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    top_k: int = 5
    candidate_k: int = 20  # how many each retriever returns before fusion
    rrf_k: int = 60  # reciprocal rank fusion constant
    rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM
    llm_provider: ProviderName = "auto"
    temperature: float = 0.1
    max_tokens: int = 700

    anthropic_api_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "DOCUSENSE_ANTHROPIC_API_KEY")
    )
    anthropic_model: str = "claude-sonnet-5-5"

    openai_api_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "DOCUSENSE_OPENAI_API_KEY")
    )
    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None  # for Azure / OpenAI-compatible servers

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
