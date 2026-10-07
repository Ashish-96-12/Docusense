"""The RAG pipeline: retrieve -> build grounded prompt -> generate -> cite."""
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Sequence

from docusense.config import Settings, get_settings
from docusense.llm import LLMError, LLMProvider, build_providers, resolve_provider
from docusense.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_message
from docusense.retrieval.embeddings import build_embedder
from docusense.retrieval.reranker import build_reranker
from docusense.schemas import (
    ChatTurn,
    Citation,
    DocumentInfo,
    ProviderStatus,
    QueryResponse,
    RetrievedChunk,
)
from docusense.store import KnowledgeBase

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
MAX_HISTORY_TURNS = 6
AUTO_CACHE_SECONDS = 60


class RAGPipeline:
    def __init__(self, settings: Optional[Settings] = None, embedder=None, reranker=None):
        self.settings = settings or get_settings()
        s = self.settings
        self.embedder = embedder or build_embedder(s.embedding_backend, s.embedding_model)
        self.reranker = reranker if reranker is not None else build_reranker(s.rerank, s.reranker_model)
        self.kb = KnowledgeBase(
            s.data_dir, self.embedder, self.reranker, s.chunk_size, s.chunk_overlap, s.candidate_k, s.rrf_k
        )
        self.providers = build_providers(s)
        self._auto_choice: Optional[LLMProvider] = None
        self._auto_checked_at = 0.0

    # ---------- documents ----------

    def ingest(self, path: Path, filename: Optional[str] = None) -> DocumentInfo:
        return self.kb.ingest(path, filename)

    def list_documents(self) -> List[DocumentInfo]:
        return self.kb.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        return self.kb.delete(doc_id)

    # ---------- retrieval ----------

    def retrieve(
        self,
        question: str,
        doc_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> List[RetrievedChunk]:
        return self.kb.retriever.retrieve(
            question,
            top_k=top_k or self.settings.top_k,
            doc_ids=doc_ids,
            mode=mode or self.settings.retrieval_mode,
            rerank=self.settings.rerank if rerank is None else rerank,
        )

    # ---------- providers ----------

    def provider_statuses(self) -> List[ProviderStatus]:
        out = []
        for name, p in self.providers.items():
            ok, detail = p.available()
            out.append(ProviderStatus(name=name, model=p.model, available=ok, detail=detail))
        return out

    def get_provider(self, name: Optional[str]) -> LLMProvider:
        name = name or self.settings.llm_provider
        if name != "auto":
            return resolve_provider(name, self.providers)
        now = time.monotonic()
        if self._auto_choice is None or now - self._auto_checked_at > AUTO_CACHE_SECONDS:
            self._auto_choice = resolve_provider("auto", self.providers)
            self._auto_checked_at = now
        return self._auto_choice

    # ---------- answering ----------

    def query(
        self,
        question: str,
        doc_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        provider: Optional[str] = None,
        history: Optional[List[ChatTurn]] = None,
    ) -> QueryResponse:
        start = time.perf_counter()
        history = (history or [])[-MAX_HISTORY_TURNS:]
        warnings: List[str] = []
        llm = self.get_provider(provider)

        # Follow-ups like "what about the second one?" retrieve badly on their
        # own, so add the previous question to the search query.
        search_query = question
        last_user = next((t.content for t in reversed(history) if t.role == "user"), None)
        if last_user and len(question.split()) < 8:
            search_query = f"{last_user} {question}"

        context = self.retrieve(search_query, doc_ids=doc_ids, top_k=top_k)
        if not context:
            answer = (
                "There are no documents to search yet. Upload one first."
                if not self.kb.chunks
                else NOT_FOUND_MESSAGE
            )
            return self._response(answer, [], llm, start, warnings)

        messages = history + [ChatTurn(role="user", content=build_user_message(question, context))]
        try:
            answer = llm.answer(
                question, context, SYSTEM_PROMPT, messages,
                temperature=self.settings.temperature, max_tokens=self.settings.max_tokens,
            )
        except LLMError as exc:
            logger.warning("Provider %s failed: %s", llm.name, exc)
            warnings.append(f"{llm.name} failed ({exc}); fell back to extractive answer.")
            llm = self.providers["extractive"]
            answer = llm.answer(question, context, SYSTEM_PROMPT, messages, temperature=0, max_tokens=0)

        if not answer.strip():
            answer = NOT_FOUND_MESSAGE
            warnings.append("The model returned an empty answer.")

        return self._response(answer, context, llm, start, warnings)

    def _response(self, answer, context, llm, start, warnings) -> QueryResponse:
        cited = {int(n) for n in _CITATION_RE.findall(answer)}
        citations = [
            Citation(
                index=n,
                doc_id=rc.chunk.doc_id,
                filename=rc.chunk.filename,
                page=rc.chunk.page,
                chunk_id=rc.chunk.chunk_id,
                score=rc.score,
                text=rc.chunk.text,
                cited=n in cited,
            )
            for n, rc in enumerate(context, start=1)
        ]
        return QueryResponse(
            answer=answer,
            citations=citations,
            provider=llm.name,
            model=llm.model,
            retrieval_mode=self.settings.retrieval_mode,
            latency_ms=int((time.perf_counter() - start) * 1000),
            warnings=warnings,
        )
