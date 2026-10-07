"""Text embedders.

The default is a sentence-transformers model. If it can't be loaded (no
internet, no torch) we fall back to a hashing embedder so the app still runs.
The hashing embedder is purely lexical, so hybrid search loses its semantic
half; the warning in the logs and /health says so.
"""
import logging
from typing import List, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized float32 vectors, shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return vecs.astype("float32")


class HashingEmbedder:
    """Offline fallback: hashed word + bigram counts, L2-normalized."""

    def __init__(self, dim: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.name = f"hashing-{dim}"
        self._vec = HashingVectorizer(
            n_features=dim, ngram_range=(1, 2), alternate_sign=False, norm="l2", stop_words="english"
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return self._vec.transform(texts).toarray().astype("float32")


def build_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # noqa: BLE001 - any load failure means fall back
        if backend == "sentence-transformers":
            raise
        logger.warning(
            "Could not load embedding model %s (%s). Falling back to the hashing embedder; "
            "semantic search quality will be lower.",
            model_name,
            exc,
        )
        return HashingEmbedder()
