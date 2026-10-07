import numpy as np

from docusense.llm.base import LLMError
from docusense.llm.extractive import extractive_answer
from docusense.llm.factory import resolve_provider
from docusense.pipeline import RAGPipeline
from docusense.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT
from docusense.retrieval.embeddings import HashingEmbedder
from docusense.schemas import ChatTurn
from tests.conftest import SAMPLES


def test_ingest_is_idempotent_and_persists(pipeline, settings):
    info1 = pipeline.ingest(SAMPLES / "docusense_info.txt")
    info2 = pipeline.ingest(SAMPLES / "docusense_info.txt")
    assert info1.doc_id == info2.doc_id
    assert len(pipeline.list_documents()) == 1

    reopened = RAGPipeline(settings, embedder=HashingEmbedder(), reranker=None)
    assert [d.doc_id for d in reopened.list_documents()] == [info1.doc_id]
    assert len(reopened.kb.chunks) == len(pipeline.kb.chunks)
    assert np.allclose(reopened.kb.embeddings, pipeline.kb.embeddings)


def test_delete_document(loaded_pipeline):
    doc = loaded_pipeline.list_documents()[0]
    assert loaded_pipeline.delete_document(doc.doc_id)
    assert all(c.doc_id != doc.doc_id for c in loaded_pipeline.kb.chunks)
    assert len(loaded_pipeline.kb.embeddings) == len(loaded_pipeline.kb.chunks)
    assert not loaded_pipeline.delete_document("nope")


def test_extractive_answer_is_grounded_and_cited(loaded_pipeline):
    res = loaded_pipeline.query("Which blockchain technologies secure EV energy transactions?")
    assert res.provider == "extractive"
    assert "Ethereum" in res.answer or "Hyperledger" in res.answer
    assert any(c.cited for c in res.citations)
    assert res.citations[0].filename == "cloud_mobility_proposal.pdf"
    assert res.citations[0].page is not None


def test_different_questions_get_different_answers(loaded_pipeline):
    a = loaded_pipeline.query("Which streaming tools process real-time vehicle data?").answer
    b = loaded_pipeline.query("What regulations must manufacturers comply with?").answer
    assert a != b


def test_short_document_is_answerable(pipeline):
    # v1 returned "No relevant information found" for every short doc
    pipeline.ingest(SAMPLES / "docusense_info.txt")
    res = pipeline.query("Why is offline processing important?")
    assert res.answer != NOT_FOUND_MESSAGE
    assert "offline" in res.answer.lower()


def test_off_topic_question_is_not_answered(loaded_pipeline):
    res = loaded_pipeline.query("What is the recipe for chocolate cake?")
    assert res.answer == NOT_FOUND_MESSAGE


def test_empty_store(pipeline):
    assert "Upload" in pipeline.query("anything").answer


def test_llm_gets_grounded_prompt_and_history(loaded_pipeline, fake_llm):
    history = [ChatTurn(role="user", content="Tell me about streaming"), ChatTurn(role="assistant", content="Sure.")]
    res = loaded_pipeline.query("Which tools?", provider="fake", history=history)

    call = fake_llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["messages"][:2] == history
    last = call["messages"][-1].content
    assert "<sources>" in last and "[1] (cloud_mobility_proposal.pdf, page" in last
    assert res.provider == "fake" and res.citations[0].cited and not res.citations[1].cited


def test_provider_failure_falls_back_to_extractive(loaded_pipeline, fake_llm):
    fake_llm.fail = True
    res = loaded_pipeline.query("Which blockchain secures EV energy?", provider="fake")
    assert res.provider == "extractive"
    assert res.warnings and "fake failed" in res.warnings[0]


def test_auto_provider_falls_back_to_extractive_without_keys(pipeline):
    assert resolve_provider("auto", pipeline.providers).name == "extractive"


def test_auto_prefers_anthropic_when_key_set(pipeline):
    pipeline.providers["anthropic"].api_key = "sk-test"
    assert resolve_provider("auto", pipeline.providers).name == "anthropic"


def test_anthropic_provider_maps_request(pipeline, monkeypatch):
    provider = pipeline.providers["anthropic"]
    provider.api_key = "sk-test"
    captured = {}

    class Block:
        type = "text"
        text = "Answer [1]."

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Resp", (), {"content": [Block()]})()

    provider._client = type("Client", (), {"messages": Messages()})()
    out = provider.complete("sys", [ChatTurn(role="user", content="q")], temperature=0.1, max_tokens=50)
    assert out == "Answer [1]."
    assert captured["system"] == "sys" and captured["messages"] == [{"role": "user", "content": "q"}]


def test_openai_provider_wraps_errors(pipeline):
    provider = pipeline.providers["openai"]
    provider.api_key = "sk-test"

    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("rate limited")

    provider._client = type("C", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    try:
        provider.complete("s", [ChatTurn(role="user", content="q")], temperature=0, max_tokens=5)
    except LLMError as exc:
        assert "rate limited" in str(exc)
    else:
        raise AssertionError("expected LLMError")


def test_ollama_unreachable_is_unavailable(pipeline):
    ok, detail = pipeline.providers["ollama"].available()
    assert not ok and "not reachable" in detail


def test_extractive_answer_with_no_overlap():
    assert extractive_answer("cake recipe", []) == NOT_FOUND_MESSAGE


def test_anthropic_retries_without_temperature(pipeline):
    provider = pipeline.providers["anthropic"]
    provider.api_key = "sk-test"
    seen = []

    class Block:
        type = "text"
        text = "ok"

    class Messages:
        def create(self, **kwargs):
            seen.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError("temperature is not supported for this model")
            return type("Resp", (), {"content": [Block()]})()

    provider._client = type("Client", (), {"messages": Messages()})()
    assert provider.complete("s", [ChatTurn(role="user", content="q")], temperature=0.1, max_tokens=5) == "ok"
    assert "temperature" in seen[0] and "temperature" not in seen[1]
