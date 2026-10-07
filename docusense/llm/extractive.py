"""Offline fallback "LLM": query-focused extractive summarization.

No model and no API key needed. It ranks the sentences of the retrieved
chunks with personalized PageRank (TextRank biased toward the question),
keeps the best few, and cites each one with its [n] source number. It can't
rephrase or reason across sources the way an LLM does, but it never makes
anything up.
"""
from typing import List

import numpy as np

from docusense.llm.base import LLMProvider
from docusense.prompts import NOT_FOUND_MESSAGE
from docusense.schemas import ChatTurn, RetrievedChunk
from docusense.text import tokenize


def _pagerank(sim: np.ndarray, personalization: np.ndarray, damping: float = 0.85, iters: int = 100) -> np.ndarray:
    n = sim.shape[0]
    row_sums = sim.sum(axis=1, keepdims=True)
    # Rows with no links jump according to the personalization vector
    transition = np.where(row_sums > 0, sim / np.where(row_sums == 0, 1, row_sums), personalization)
    scores = np.full(n, 1.0 / n)
    for _ in range(iters):
        new = (1 - damping) * personalization + damping * scores @ transition
        if np.abs(new - scores).sum() < 1e-8:
            break
        scores = new
    return scores


def extractive_answer(question: str, context: List[RetrievedChunk], max_sentences: int = 3) -> str:
    q_terms = set(tokenize(question))
    sentences, sources, token_sets = [], [], []
    for n, rc in enumerate(context, start=1):
        # Chunks store one sentence per line (see chunking.py)
        for s in rc.chunk.text.split("\n"):
            s = s.strip()
            toks = set(tokenize(s))
            if len(toks) < 3:
                continue
            if s in sentences:  # overlapping chunks repeat sentences
                continue
            sentences.append(s)
            sources.append(n)
            token_sets.append(toks)

    if not sentences or not q_terms:
        return NOT_FOUND_MESSAGE
    overlap = np.array([len(q_terms & t) for t in token_sets], dtype=float)
    if overlap.max() == 0:
        return NOT_FOUND_MESSAGE

    # Cosine-style similarity between sentences on shared words
    m = len(sentences)
    sim = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            shared = len(token_sets[i] & token_sets[j])
            if shared:
                sim[i, j] = sim[j, i] = shared / np.sqrt(len(token_sets[i]) * len(token_sets[j]))

    personalization = overlap + 0.05
    personalization /= personalization.sum()
    scores = _pagerank(sim, personalization)
    # Lean on question overlap heavily so the answer stays on topic;
    # also prefer sentences from higher-ranked chunks.
    rank_bonus = np.array([1.0 / (1 + 0.15 * (src - 1)) for src in sources])
    final = scores * (1 + 3 * overlap / overlap.max()) * rank_bonus
    final[overlap == 0] *= 0.1

    picked = sorted(np.argsort(-final)[:max_sentences])
    return " ".join(f"{sentences[i]} [{sources[i]}]" for i in picked)


class ExtractiveProvider(LLMProvider):
    name = "extractive"
    model = "query-focused-textrank"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def available(self) -> tuple[bool, str]:
        return True, "always available (no LLM, extracts sentences from the documents)"

    def complete(self, system: str, messages: List[ChatTurn], *, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError("The extractive provider answers from retrieved context only")

    def answer(self, question, context, system, messages, *, temperature, max_tokens) -> str:
        return extractive_answer(question, context, self.max_sentences)
