"""Sentence-aware chunking.

Sentences are packed into chunks of about `chunk_size` words. Each new chunk
starts with the last ~`chunk_overlap` words of sentences from the previous
one, so an answer that straddles a boundary is still retrievable. Chunks never
cross a page, so every chunk can cite one page number.
"""
from typing import List

from docusense.ingestion.loaders import Page
from docusense.schemas import Chunk
from docusense.text import split_sentences


def _split_long_sentence(sentence: str, max_words: int) -> List[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_pages(
    pages: List[Page], doc_id: str, filename: str, chunk_size: int = 220, chunk_overlap: int = 40
) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    chunks: List[Chunk] = []

    def flush(sentences: List[str], page_number):
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{len(chunks):04d}",
                doc_id=doc_id,
                filename=filename,
                text="\n".join(sentences),  # one sentence per line
                page=page_number,
                position=len(chunks),
            )
        )

    for page in pages:
        sentences: List[str] = []
        for s in split_sentences(page.text):
            sentences.extend(_split_long_sentence(s, chunk_size) if len(s.split()) > chunk_size else [s])

        current: List[str] = []
        current_words = 0
        new_since_flush = False


        for sentence in sentences:
            n = len(sentence.split())
            if current and current_words + n > chunk_size:
                flush(current, page.number)
                # Carry trailing sentences forward as overlap
                carried, carried_words = [], 0
                for prev in reversed(current):
                    w = len(prev.split())
                    if carried_words + w > chunk_overlap:
                        break
                    carried.insert(0, prev)
                    carried_words += w
                current, current_words = carried, carried_words
                new_since_flush = False
            current.append(sentence)
            current_words += n
            new_since_flush = True

        if current and new_since_flush:
            flush(current, page.number)

    return chunks
