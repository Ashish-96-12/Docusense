"""Hybrid retriever: BM25 + dense vectors, fused with RRF, optionally reranked.

    question ─┬─> BM25 (keywords) ──────┐
              └─> embed -> FAISS (meaning) ┴─> RRF fusion -> cross-encoder rerank -> top_k

BM25 catches exact terms (product names, IDs, acronyms); embeddings catch
paraphrases ("cars selling power back to the grid" -> "vehicle-to-grid").
"""
from typing import List, Optional, Sequence, Set

import numpy as np

from docusense.retrieval.bm25 import BM25Index
from docusense.retrieval.embeddings import Embedder
from docusense.retrieval.fusion import reciprocal_rank_fusion
from docusense.retrieval.reranker import CrossEncoderReranker
from docusense.retrieval.vector_store import VectorIndex
from docusense.schemas import Chunk, RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        embedder: Embedder,
        reranker: Optional[CrossEncoderReranker] = None,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must line up one-to-one")
        self.chunks = list(chunks)
        self.embedder = embedder
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._bm25 = BM25Index([c.text for c in self.chunks])
        self._vectors = VectorIndex(embeddings, embedder.dim)

    def _allowed_rows(self, doc_ids: Optional[Sequence[str]]) -> Optional[Set[int]]:
        if not doc_ids:
            return None
        wanted = set(doc_ids)
        return {i for i, c in enumerate(self.chunks) if c.doc_id in wanted}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[Sequence[str]] = None,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> List[RetrievedChunk]:
        if not self.chunks:
            return []
        allowed = self._allowed_rows(doc_ids)
        if allowed is not None and not allowed:
            return []

        k = max(self.candidate_k, top_k)
        bm25_hits = self._bm25.search(query, k, allowed) if mode in ("hybrid", "bm25") else []
        vector_hits = []
        if mode in ("hybrid", "vector"):
            qvec = self.embedder.embed([query])[0]
            vector_hits = self._vectors.search(qvec, k, allowed)

        bm25_rank = {row: r for r, (row, _) in enumerate(bm25_hits, start=1)}
        vector_rank = {row: r for r, (row, _) in enumerate(vector_hits, start=1)}

        if mode == "bm25":
            ranked = [(row, score) for row, score in bm25_hits]
        elif mode == "vector":
            ranked = [(row, score) for row, score in vector_hits]
        else:
            ranked = reciprocal_rank_fusion(
                [[r for r, _ in bm25_hits], [r for r, _ in vector_hits]], k=self.rrf_k
            )

        shortlist = ranked[: max(top_k * 3, top_k)] if (rerank and self.reranker) else ranked[:top_k]
        if rerank and self.reranker and shortlist:
            scores = self.reranker.score(query, [self.chunks[row].text for row, _ in shortlist])
            rows = [row for row, _ in shortlist]
            shortlist = sorted(zip(rows, scores, strict=True), key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(
                chunk=self.chunks[row],
                score=float(score),
                bm25_rank=bm25_rank.get(row),
                vector_rank=vector_rank.get(row),
            )
            for row, score in shortlist[:top_k]
        ]
