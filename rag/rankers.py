from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np
from .utils import clean_text


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []

    def index_chunks(self, chunks: List[Dict]):
        """Index chunks for BM25 retrieval"""
        self.chunks = chunks

        # Tokenize chunks for BM25
        self.tokenized_chunks = []
        for chunk in chunks:
            clean_chunk = clean_text(chunk['text'])
            tokens = clean_chunk.lower().split()
            self.tokenized_chunks.append(tokens)

        # Create BM25 index
        if self.tokenized_chunks:
            self.bm25 = BM25Okapi(self.tokenized_chunks)

        print(f"✅ Indexed {len(chunks)} chunks for BM25 retrieval")

    def search(self, query: str, top_k: int = 6) -> List[Dict]:
        """Search for relevant chunks"""
        if not self.bm25 or not self.chunks:
            return []

        # Tokenize query
        query_tokens = clean_text(query).lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k results (removed score threshold)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            # Include all results, even with score 0 for debugging
            result = self.chunks[idx].copy()
            result['score'] = float(scores[idx])
            result['rank'] = len(results) + 1
            results.append(result)
            print(f"Debug: Found chunk with score {scores[idx]:.4f}: {self.chunks[idx]['text'][:100]}...")

        return results

    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        return {
            "total_chunks": len(self.chunks),
            "indexed": self.bm25 is not None,
            "avg_chunk_length": np.mean([len(c['text']) for c in self.chunks]) if self.chunks else 0
        }