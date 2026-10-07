"""Shared fixtures. Everything here runs offline: hashing embeddings, no reranker,
and a fake LLM provider, so the suite is fast and needs no API keys."""
from pathlib import Path
from typing import List

import pytest

from docusense.config import Settings
from docusense.llm.base import LLMError, LLMProvider
from docusense.pipeline import RAGPipeline
from docusense.retrieval.embeddings import HashingEmbedder
from docusense.schemas import ChatTurn

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


class FakeLLM(LLMProvider):
    """Records what it was asked and replies with a canned, cited answer."""

    name = "fake"
    model = "fake-1"

    def __init__(self, reply: str = "Kafka and AWS Kinesis stream vehicle data [1].", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.calls: List[dict] = []

    def available(self):
        return True, "fake"

    def complete(self, system: str, messages: List[ChatTurn], *, temperature: float, max_tokens: int) -> str:
        self.calls.append({"system": system, "messages": messages})
        if self.fail:
            raise LLMError("boom")
        return self.reply


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "store",
        embedding_backend="hashing",
        rerank=False,
        llm_provider="extractive",
        anthropic_api_key=None,
        openai_api_key=None,
        ollama_host="http://127.0.0.1:9",  # nothing listens here
    )


@pytest.fixture
def pipeline(settings) -> RAGPipeline:
    return RAGPipeline(settings, embedder=HashingEmbedder(), reranker=None)


@pytest.fixture
def loaded_pipeline(pipeline) -> RAGPipeline:
    for f in ["cloud_mobility_proposal.pdf", "docusense_info.txt"]:
        pipeline.ingest(SAMPLES / f)
    return pipeline


@pytest.fixture
def fake_llm(pipeline) -> FakeLLM:
    fake = FakeLLM()
    pipeline.providers["fake"] = fake
    return fake
