"""Reciprocal Rank Fusion (Cormack et al., 2009).

Combines ranked lists using only ranks, so BM25 scores and cosine
similarities never need to be put on the same scale.
"""
from typing import Dict, List, Sequence, Tuple


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]], k: int = 60
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
