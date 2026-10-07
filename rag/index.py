import os
import json
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "for", "and", "or", "it", "this", "that", "what", "how", "why", "does", "do",
    "with", "as", "by", "at", "from", "about",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation and drop very common words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class IndexManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.indices = {}  # Separate index per document
        self.chunks_store = {}  # Store chunks per document
        self.load_indices()

    def load_indices(self):
        """Load existing indices from disk"""
        index_dir = "storage/indices"
        if os.path.exists(index_dir):
            for filename in os.listdir(index_dir):
                if filename.endswith(".pkl"):
                    doc_id = filename[:-4]
                    with open(os.path.join(index_dir, filename), 'rb') as f:
                        self.indices[doc_id] = pickle.load(f)

        chunks_dir = "storage/chunks"
        if os.path.exists(chunks_dir):
            for filename in os.listdir(chunks_dir):
                if filename.endswith(".json"):
                    doc_id = filename[:-5]
                    with open(os.path.join(chunks_dir, filename), 'r') as f:
                        self.chunks_store[doc_id] = json.load(f)

    def save_index(self, doc_id: str):
        """Save index for a specific document"""
        os.makedirs("storage/indices", exist_ok=True)
        os.makedirs("storage/chunks", exist_ok=True)

        if doc_id in self.indices:
            with open(f"storage/indices/{doc_id}.pkl", 'wb') as f:
                pickle.dump(self.indices[doc_id], f)

        if doc_id in self.chunks_store:
            with open(f"storage/chunks/{doc_id}.json", 'w') as f:
                json.dump(self.chunks_store[doc_id], f, indent=2)

    def add(self, doc_id: str, chunks: List[Dict[str, Any]]):
        """Add chunks for a specific document"""
        # Store chunks with document ID
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {})
            }
            enhanced_chunks.append(enhanced_chunk)

        self.chunks_store[doc_id] = enhanced_chunks

        # Create BM25 index for this document
        tokenized_chunks = [tokenize(chunk["text"]) for chunk in enhanced_chunks]
        self.indices[doc_id] = BM25Okapi(tokenized_chunks)

        # Save to disk
        self.save_index(doc_id)

    def retrieve(self, query: str, doc_id: str = None, top_k: int = 6) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks, optionally filtered by document"""
        if doc_id and doc_id in self.indices:
            # Search within specific document
            return self._search_document(query, doc_id, top_k)
        elif doc_id:
            # Document not found
            return []
        else:
            # Search across all documents
            return self._search_all_documents(query, top_k)

    def _search_document(self, query: str, doc_id: str, top_k: int) -> List[Dict[str, Any]]:
        """Search within a specific document"""
        if doc_id not in self.indices or doc_id not in self.chunks_store:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        bm25 = self.indices[doc_id]
        chunks = self.chunks_store[doc_id]

        # Get BM25 scores
        scores = np.asarray(bm25.get_scores(tokenized_query), dtype=float)

        # BM25 IDF goes to zero or negative when a document has very few
        # chunks (a short doc becomes one chunk), so nothing would ever score
        # above 0. Fall back to simple query-term overlap in that case.
        if not np.any(scores > 0):
            query_terms = set(tokenized_query)
            scores = np.array([
                len(query_terms & set(tokenize(c["text"]))) / len(query_terms)
                for c in chunks
            ], dtype=float)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return chunks with scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include chunks with positive scores
                chunk = chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                results.append(chunk)

        return results

    def _search_all_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search across all documents"""
        all_results = []

        for doc_id in self.indices.keys():
            doc_results = self._search_document(query, doc_id, top_k)
            all_results.extend(doc_results)

        # Sort by score and return top-k
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k]

    def remove_document(self, doc_id: str):
        """Remove a document from the index"""
        if doc_id in self.indices:
            del self.indices[doc_id]
        if doc_id in self.chunks_store:
            del self.chunks_store[doc_id]

        # Remove from disk
        index_path = f"storage/indices/{doc_id}.pkl"
        chunks_path = f"storage/chunks/{doc_id}.json"

        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(chunks_path):
            os.remove(chunks_path)