"""Pydantic models shared by the pipeline and the API."""
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    page: Optional[int] = None  # 1-based page number for PDFs
    position: int  # chunk order within the document


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    num_pages: Optional[int] = None
    num_words: int
    ingested_at: datetime


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = Field(description="Final ranking score (RRF or reranker)")
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None


class Citation(BaseModel):
    index: int = Field(description="The [n] number used in the answer")
    doc_id: str
    filename: str
    page: Optional[int] = None
    chunk_id: str
    score: float
    text: str
    cited: bool = Field(description="True if the answer actually references this source")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    doc_ids: Optional[List[str]] = Field(default=None, description="Limit search to these documents")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    provider: Optional[Literal["auto", "anthropic", "openai", "ollama", "extractive"]] = None
    history: List[ChatTurn] = Field(default_factory=list, description="Earlier turns, for follow-ups")


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    provider: str
    model: str
    retrieval_mode: str
    latency_ms: int
    warnings: List[str] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    doc_ids: Optional[List[str]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    mode: Optional[Literal["hybrid", "vector", "bm25"]] = None


class ProviderStatus(BaseModel):
    name: str
    model: str
    available: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    documents: int
    chunks: int
    embedder: str
    reranker: Optional[str]
    default_provider: str
