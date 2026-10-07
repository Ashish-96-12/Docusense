"""Dense vector search with FAISS (inner product on normalized vectors = cosine)."""
from typing import List, Optional, Set, Tuple

import numpy as np


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, dim: int):
        import faiss

        self._n = len(embeddings)
        self._index = faiss.IndexFlatIP(dim)
        if self._n:
            self._index.add(np.ascontiguousarray(embeddings, dtype="float32"))

    def search(
        self, query_vec: np.ndarray, k: int, allowed: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        if self._n == 0:
            return []
        # When filtering to some documents, over-fetch so enough survive the filter
        fetch = self._n if allowed is not None else min(k, self._n)
        scores, ids = self._index.search(np.ascontiguousarray(query_vec.reshape(1, -1), dtype="float32"), fetch)
        hits = []
        for i, s in zip(ids[0], scores[0], strict=True):
            if i < 0 or (allowed is not None and i not in allowed):
                continue
            hits.append((int(i), float(s)))
            if len(hits) == k:
                break
        return hits
