import pytest
from fastapi.testclient import TestClient

from docusense.api import app
from tests.conftest import SAMPLES


@pytest.fixture
def client(pipeline):
    app.state.pipeline = pipeline
    with TestClient(app) as c:
        yield c
    del app.state.pipeline


def upload(client, name):
    with open(SAMPLES / name, "rb") as f:
        return client.post("/documents", files={"file": (name, f)})


def test_health_and_providers(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["documents"] == 0 and h["embedder"].startswith("hashing")
    names = {p["name"] for p in client.get("/providers").json()}
    assert names == {"anthropic", "openai", "ollama", "extractive"}


def test_upload_list_query_delete(client):
    r = upload(client, "cloud_mobility_proposal.pdf")
    assert r.status_code == 201
    doc = r.json()
    assert doc["num_pages"] and doc["num_chunks"] > 5

    assert [d["doc_id"] for d in client.get("/documents").json()] == [doc["doc_id"]]

    q = client.post("/query", json={"question": "Which edge AI hardware helps vehicles decide quickly?"})
    assert q.status_code == 200
    body = q.json()
    assert "Jetson" in body["answer"]
    assert body["citations"] and body["provider"] == "extractive"

    r = client.post("/retrieve", json={"question": "GDPR CCPA", "top_k": 2, "mode": "bm25"})
    assert r.status_code == 200 and len(r.json()) == 2

    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 404


def test_upload_validation(client):
    r = client.post("/documents", files={"file": ("data.csv", b"a,b")})
    assert r.status_code == 415
    r = client.post("/documents", files={"file": ("blank.txt", b"   ")})
    assert r.status_code == 422


def test_upload_size_limit(client, pipeline):
    pipeline.settings.max_upload_mb = 0
    r = client.post("/documents", files={"file": ("big.txt", b"x" * 2048)})
    assert r.status_code == 413


def test_query_validation(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "hi", "doc_ids": ["missing"]}).status_code == 404
    assert client.post("/query", json={"question": "hi", "provider": "gpt9"}).status_code == 422
