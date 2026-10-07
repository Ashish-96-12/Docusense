import numpy as np

from docusense.retrieval.bm25 import BM25Index
from docusense.retrieval.embeddings import HashingEmbedder
from docusense.retrieval.fusion import reciprocal_rank_fusion
from docusense.retrieval.hybrid import HybridRetriever
from docusense.schemas import Chunk

TEXTS = [
    "Kafka, AWS Kinesis and Google Pub/Sub stream real-time vehicle data.",
    "Blockchain such as Ethereum and Hyperledger Fabric secures EV energy trades.",
    "NVIDIA Jetson lets vehicles make quick decisions at the edge.",
    "GDPR and CCPA require strong data privacy controls.",
]


def make_chunks(texts, doc_ids=None):
    doc_ids = doc_ids or ["d1"] * len(texts)
    return [
        Chunk(chunk_id=f"c{i}", doc_id=d, filename=f"{d}.txt", text=t, position=i)
        for i, (t, d) in enumerate(zip(texts, doc_ids, strict=True))
    ]


def test_bm25_single_chunk_still_matches():
    # The v1 bug: one chunk -> negative IDF -> nothing ever returned
    idx = BM25Index(["DocuSense is an offline document analysis system."])
    assert idx.search("What is DocuSense?", k=3) == [(0, 1.0)]
    assert idx.search("unrelated banana", k=3) == []


def test_bm25_ranks_keyword_match_first():
    hits = BM25Index(TEXTS).search("Which blockchain secures energy?", k=4)
    assert hits[0][0] == 1


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    order = [item for item, _ in fused]
    assert order[0] == 1  # ranked 1st and 2nd beats 1st-only or 3rd+1st
    assert set(order) == {1, 2, 3, 4}


def test_hybrid_retriever_modes_and_ranks():
    emb = HashingEmbedder()
    r = HybridRetriever(make_chunks(TEXTS), emb.embed(TEXTS), emb)
    for mode in ("hybrid", "bm25", "vector"):
        hits = r.retrieve("edge decisions with NVIDIA Jetson", top_k=2, mode=mode)
        assert hits[0].chunk.chunk_id == "c2", mode
    hybrid_top = r.retrieve("edge decisions with NVIDIA Jetson", top_k=1)[0]
    assert hybrid_top.bm25_rank == 1 and hybrid_top.vector_rank == 1


def test_hybrid_filters_by_document():
    emb = HashingEmbedder()
    chunks = make_chunks(TEXTS, ["a", "a", "b", "b"])
    r = HybridRetriever(chunks, emb.embed(TEXTS), emb)
    hits = r.retrieve("Kafka streaming", top_k=4, doc_ids=["b"])
    assert hits and all(h.chunk.doc_id == "b" for h in hits)
    assert r.retrieve("Kafka", doc_ids=["missing"]) == []


def test_reranker_reorders_shortlist():
    class ReverseReranker:
        name = "reverse"

        def score(self, query, passages):
            return [float(i) for i in range(len(passages))]  # last passage wins

    emb = HashingEmbedder()
    r = HybridRetriever(make_chunks(TEXTS), emb.embed(TEXTS), emb, reranker=ReverseReranker())
    without = r.retrieve("vehicle data privacy", top_k=2, rerank=False)
    with_rr = r.retrieve("vehicle data privacy", top_k=2, rerank=True)
    assert [h.chunk.chunk_id for h in without] != [h.chunk.chunk_id for h in with_rr]


def test_hashing_embedder_is_normalized():
    vecs = HashingEmbedder().embed(["hello world", "another sentence here"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
