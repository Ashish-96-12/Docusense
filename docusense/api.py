"""FastAPI backend.

Run with:  uvicorn docusense.api:app --reload
Docs at:   http://localhost:8000/docs
"""
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from docusense import __version__
from docusense.ingestion.loaders import SUPPORTED_EXTENSIONS, EmptyDocument, UnsupportedFileType
from docusense.pipeline import RAGPipeline
from docusense.schemas import (
    DocumentInfo,
    HealthResponse,
    ProviderStatus,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    RetrieveRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests can inject a ready-made pipeline before startup
    if not hasattr(app.state, "pipeline"):
        app.state.pipeline = RAGPipeline()
    yield


app = FastAPI(
    title="DocuSense",
    version=__version__,
    description="Upload documents and ask questions. Hybrid BM25 + vector retrieval with cited LLM answers.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


@app.get("/health", response_model=HealthResponse)
def health(pipe: RAGPipeline = Depends(get_pipeline)):
    return HealthResponse(
        documents=len(pipe.kb.documents),
        chunks=len(pipe.kb.chunks),
        embedder=pipe.embedder.name,
        reranker=pipe.reranker.name if pipe.reranker else None,
        default_provider=pipe.get_provider(None).name,
    )


@app.get("/providers", response_model=List[ProviderStatus])
def providers(pipe: RAGPipeline = Depends(get_pipeline)):
    return pipe.provider_statuses()


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents(pipe: RAGPipeline = Depends(get_pipeline)):
    return pipe.list_documents()


@app.post("/documents", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...), pipe: RAGPipeline = Depends(get_pipeline)):
    filename = Path(file.filename or "upload").name  # drop any client-supplied path
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Use one of {sorted(SUPPORTED_EXTENSIONS)}")

    limit = pipe.settings.max_upload_mb * 1024 * 1024
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / f"upload{suffix}"
        size = 0
        with open(tmp_path, "wb") as out:
            while block := await file.read(1 << 20):
                size += len(block)
                if size > limit:
                    raise HTTPException(413, f"File is larger than {pipe.settings.max_upload_mb} MB")
                out.write(block)
        try:
            return await run_in_threadpool(pipe.ingest, tmp_path, filename)
        except (UnsupportedFileType, EmptyDocument) as exc:
            raise HTTPException(422, str(exc)) from exc


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, pipe: RAGPipeline = Depends(get_pipeline)):
    if not pipe.delete_document(doc_id):
        raise HTTPException(404, "Document not found")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, pipe: RAGPipeline = Depends(get_pipeline)):
    _check_doc_ids(pipe, req.doc_ids)
    try:
        return await run_in_threadpool(pipe.query, req.question, req.doc_ids, req.top_k, req.provider, req.history)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/retrieve", response_model=List[RetrievedChunk])
async def retrieve(req: RetrieveRequest, pipe: RAGPipeline = Depends(get_pipeline)):
    """Retrieval only, no LLM. Handy for debugging and evaluation."""
    _check_doc_ids(pipe, req.doc_ids)
    return await run_in_threadpool(pipe.retrieve, req.question, req.doc_ids, req.top_k, req.mode)


def _check_doc_ids(pipe: RAGPipeline, doc_ids):
    unknown = [d for d in (doc_ids or []) if d not in pipe.kb.documents]
    if unknown:
        raise HTTPException(404, f"Unknown document ids: {unknown}")
