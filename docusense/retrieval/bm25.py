"""Keyword search with BM25."""
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from docusense.text import tokenize


class BM25Index:
    def __init__(self, texts: Sequence[str]):
        self._tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokens) if any(self._tokens) else None

    def search(
        self, query: str, k: int, allowed: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Return (row, score) pairs for rows with a positive match, best first."""
        q = tokenize(query)
        if not q or self._bm25 is None:
            return []

        scores = np.asarray(self._bm25.get_scores(q), dtype=float)

        # With very few chunks BM25's IDF goes to zero or negative, so a real
        # match can score <= 0. Fall back to query-term overlap then.
        if not np.any(scores > 0):
            qset = set(q)
            scores = np.array([len(qset.intersection(toks)) / len(qset) for toks in self._tokens])

        rows = range(len(scores)) if allowed is None else allowed
        hits = [(i, float(scores[i])) for i in rows if scores[i] > 0]
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]
