"""Optional cross-encoder reranker.

A cross-encoder reads the question and a passage together, so it judges
relevance much better than either BM25 or embeddings alone. It's slower, so
we only run it on the fused shortlist.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device="cpu")
        self.name = model_name

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.predict([(query, p) for p in passages], show_progress_bar=False)]


def build_reranker(enabled: bool, model_name: str) -> Optional[CrossEncoderReranker]:
    if not enabled:
        return None
    try:
        return CrossEncoderReranker(model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranker %s unavailable (%s); using fused ranking only.", model_name, exc)
        return None
