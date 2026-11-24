"""Chunking strategies.

Every chunk stays inside a single page, so a citation always names the exact
page the text came from. Two strategies are provided so they can be compared
in the evaluation harness:

* ``fixed``    - sliding window of N words with M words of overlap.
* ``sentence`` - pack whole sentences up to N words, carrying the last M
                 sentences into the next chunk as overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .loader import Page


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    doc_name: str
    page: int
    index: int  # position of the chunk within its page
    text: str


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[$])")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _word_count(s: str) -> int:
    return len(s.split())


def chunk_fixed(page: Page, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size - 1))
    words = page.text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        out.append(" ".join(window))
        if start + size >= len(words):
            break
    return out


def chunk_sentence(page: Page, size: int, overlap: int) -> list[str]:
    """`overlap` is a number of sentences here, not words."""
    if size <= 0:
        raise ValueError("size must be positive")
    sentences = split_sentences(page.text)
    chunks: list[list[str]] = []
    current: list[str] = []
    count = 0
    for sent in sentences:
        n = _word_count(sent)
        if current and count + n > size:
            chunks.append(current)
            current = current[-overlap:] if overlap > 0 else []
            count = sum(_word_count(s) for s in current)
        current.append(sent)
        count += n
    if current and (not chunks or current != chunks[-1][-len(current):]):
        chunks.append(current)
    return [" ".join(c) for c in chunks]


STRATEGIES: dict[str, Callable[[Page, int, int], list[str]]] = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
}


def chunk_pages(
    doc_id: str,
    doc_name: str,
    pages: list[Page],
    strategy: str = "sentence",
    size: int = 120,
    overlap: int = 1,
) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunker '{strategy}'. Options: {sorted(STRATEGIES)}")
    fn = STRATEGIES[strategy]
    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(fn(page, size, overlap)):
            if text.strip():
                chunks.append(Chunk(doc_id, doc_name, page.number, i, text))
    return chunks
