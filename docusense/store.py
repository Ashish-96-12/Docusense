"""Persistent document store.

Everything lives under settings.data_dir:
    documents.json   document metadata
    chunks.jsonl     one chunk per line, same order as embeddings
    embeddings.npy   float32 matrix, one row per chunk
    meta.json        which embedder produced embeddings.npy

The BM25 and FAISS indexes are rebuilt in memory on load and after every
change. That's fast up to tens of thousands of chunks; past that you'd swap
in a vector database.
"""
import hashlib
import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from docusense.ingestion.chunking import chunk_pages
from docusense.ingestion.loaders import load_document
from docusense.retrieval.embeddings import Embedder
from docusense.retrieval.hybrid import HybridRetriever
from docusense.retrieval.reranker import CrossEncoderReranker
from docusense.schemas import Chunk, DocumentInfo

logger = logging.getLogger(__name__)


def file_doc_id(path: Path) -> str:
    """Content hash, so re-uploading the same file doesn't duplicate it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


class KnowledgeBase:
    def __init__(
        self,
        data_dir: Path,
        embedder: Embedder,
        reranker: Optional[CrossEncoderReranker] = None,
        chunk_size: int = 220,
        chunk_overlap: int = 40,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        self.data_dir = Path(data_dir)
        self.files_dir = self.data_dir / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.reranker = reranker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

        self._lock = threading.RLock()
        self.documents: Dict[str, DocumentInfo] = {}
        self.chunks: List[Chunk] = []
        self.embeddings = np.zeros((0, embedder.dim), dtype="float32")
        self._load()
        self._rebuild()

    # ---------- persistence ----------

    def _load(self) -> None:
        docs_path = self.data_dir / "documents.json"
        if not docs_path.exists():
            return
        self.documents = {
            d["doc_id"]: DocumentInfo(**d) for d in json.loads(docs_path.read_text(encoding="utf-8"))
        }
        with open(self.data_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]

        meta_path = self.data_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        emb_path = self.data_dir / "embeddings.npy"
        if meta.get("embedder") == self.embedder.name and emb_path.exists():
            self.embeddings = np.load(emb_path)
        else:
            logger.info("Embedder changed (%s -> %s); re-embedding %d chunks", meta.get("embedder"),
                        self.embedder.name, len(self.chunks))
            self.embeddings = self.embedder.embed([c.text for c in self.chunks])
            self._save()

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.data_dir / ".tmp"
        tmp.mkdir(exist_ok=True)
        (tmp / "documents.json").write_text(
            json.dumps([d.model_dump(mode="json") for d in self.documents.values()], indent=2), encoding="utf-8"
        )
        with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        np.save(tmp / "embeddings.npy", self.embeddings)
        (tmp / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "dim": self.embedder.dim}))
        for p in tmp.iterdir():  # swap in all files together
            p.replace(self.data_dir / p.name)
        tmp.rmdir()

    def _rebuild(self) -> None:
        self.retriever = HybridRetriever(
            self.chunks, self.embeddings, self.embedder, self.reranker, self.candidate_k, self.rrf_k
        )

    # ---------- public API ----------

    def ingest(self, path: Path, filename: Optional[str] = None) -> DocumentInfo:
        path = Path(path)
        filename = filename or path.name
        doc_id = file_doc_id(path)
        with self._lock:
            if doc_id in self.documents:
                return self.documents[doc_id]

        pages = load_document(path)  # raises on unsupported/empty files
        chunks = chunk_pages(pages, doc_id, filename, self.chunk_size, self.chunk_overlap)
        vectors = self.embedder.embed([c.text for c in chunks])

        with self._lock:
            if doc_id in self.documents:  # someone else finished first
                return self.documents[doc_id]
            stored = self.files_dir / f"{doc_id}{path.suffix.lower()}"
            if path.resolve() != stored.resolve():
                shutil.copyfile(path, stored)
            info = DocumentInfo(
                doc_id=doc_id,
                filename=filename,
                num_chunks=len(chunks),
                num_pages=len(pages) if pages[0].number is not None else None,
                num_words=sum(len(p.text.split()) for p in pages),
                ingested_at=datetime.now(timezone.utc),
            )
            self.documents[doc_id] = info
            self.chunks.extend(chunks)
            self.embeddings = np.vstack([self.embeddings, vectors]) if len(vectors) else self.embeddings
            self._save()
            self._rebuild()
        logger.info("Ingested %s as %s (%d chunks)", filename, doc_id, len(chunks))
        return info

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.documents:
                return False
            keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            self.embeddings = self.embeddings[keep] if keep else np.zeros((0, self.embedder.dim), dtype="float32")
            del self.documents[doc_id]
            for f in self.files_dir.glob(f"{doc_id}.*"):
                f.unlink()
            self._save()
            self._rebuild()
        return True

    def list_documents(self) -> List[DocumentInfo]:
        return sorted(self.documents.values(), key=lambda d: d.ingested_at)
