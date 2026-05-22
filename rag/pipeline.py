import hashlib
import json
import os
from datetime import datetime
from typing import List, Dict, Any


class RagPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        from .index import IndexManager
        self.idx = IndexManager(cfg)
        self.document_metadata = {}
        self.load_metadata()

    def load_metadata(self):
        """Load document metadata from storage"""
        metadata_path = "storage/metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.document_metadata = json.load(f)

    def save_metadata(self):
        """Save document metadata to storage"""
        metadata_path = "storage/metadata.json"
        os.makedirs("storage", exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(self.document_metadata, f, indent=2)

    def ingest(self, path: str) -> str:
        """Ingest a document and return its unique ID"""
        from .chunkers import parse_and_chunk
        from .utils import uid

        # Generate unique document ID
        doc_id = uid(path)

        # Parse and chunk the document
        chunks = parse_and_chunk(path, self.cfg.get("chunking", {}))

        # Store metadata
        self.document_metadata[doc_id] = {
            "path": path,
            "filename": os.path.basename(path),
            "ingested_at": datetime.now().isoformat(),
            "num_chunks": len(chunks)
        }
        self.save_metadata()

        # Add to index with document ID
        self.idx.add(doc_id, chunks)

        return doc_id

    def answer(self, query: str, doc_id: str = None, top_k: int = 6):
        """Answer a query, optionally filtering by document ID"""
        from .summarizer import textrank_summary

        # Retrieve relevant chunks
        hits = self.idx.retrieve(query, doc_id=doc_id, top_k=top_k)

        if not hits:
            return "No relevant information found for your query.", []

        # Combine chunks into context
        context = "\n\n".join(h["text"] for h in hits)

        # Generate answer using TextRank summarization
        answer = textrank_summary(context, max_sentences=3)

        return answer, hits

    def clear_document(self, doc_id: str):
        """Clear a specific document from the index"""
        self.idx.remove_document(doc_id)
        if doc_id in self.document_metadata:
            del self.document_metadata[doc_id]
            self.save_metadata()