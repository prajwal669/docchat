"""The RAG pipeline: ingest documents, retrieve passages, answer with citations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_pages
from .config import Settings, get_settings
from .llm import NOT_FOUND, Answerer, Context, ExtractiveAnswerer, LLMError, get_answerer
from .loader import Page, load_document, safe_name
from .retrieval import (
    BM25Retriever,
    Retriever,
    RetrieverUnavailable,
    build_retriever,
    rank_top_k,
    tokenize,
)

log = logging.getLogger("docchat")
STORE_VERSION = 1


@dataclass
class DocRecord:
    id: str
    name: str
    sha256: str
    added_at: float
    pages: list[Page]
    n_chunks: int = 0


@dataclass
class Citation:
    n: int
    doc_id: str
    doc_name: str
    page: int
    score: float
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class AnswerResult:
    answer: str
    abstained: bool
    citations: list[Citation]
    retriever: str
    answerer: str
    latency_ms: float
    coverage: float = 0.0
    warning: str | None = None


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        answerer: Answerer | None = None,
        persist: bool = True,
    ):
        self.settings = settings or get_settings()
        self.persist = persist
        self._lock = threading.RLock()
        self._docs: dict[str, DocRecord] = {}
        self._chunks: list[Chunk] = []
        self._chunk_terms: list[frozenset[str]] = []
        self._indexes: dict[str, Retriever] = {}
        self._custom_answerer = answerer
        self._store_path = self.settings.storage_dir / "store.json"
        if persist:
            self._load_store()
        self._rebuild()

    # ------------------------------------------------------------------ store

    def _load_store(self) -> None:
        if not self._store_path.exists():
            return
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
            for d in raw["docs"]:
                pages = [Page(p["number"], p["text"]) for p in d["pages"]]
                self._docs[d["id"]] = DocRecord(d["id"], d["name"], d["sha256"], d["added_at"], pages)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            backup = self._store_path.with_suffix(".corrupt")
            log.warning("Could not read %s (%s); moved to %s and starting empty", self._store_path, exc, backup)
            try:
                self._store_path.replace(backup)
            except OSError:
                pass
            self._docs = {}

    def _save_store(self) -> None:
        if not self.persist:
            return
        self.settings.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STORE_VERSION,
            "docs": [
                {
                    "id": d.id, "name": d.name, "sha256": d.sha256, "added_at": d.added_at,
                    "pages": [{"number": p.number, "text": p.text} for p in d.pages],
                }
                for d in self._docs.values()
            ],
        }
        # Atomic write: never leave a half-written store behind.
        fd, tmp = tempfile.mkstemp(dir=self.settings.storage_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._store_path)
        except OSError:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _rebuild(self) -> None:
        s = self.settings
        chunks: list[Chunk] = []
        for doc in self._docs.values():
            doc_chunks = chunk_pages(doc.id, doc.name, doc.pages, s.chunker, s.chunk_size, s.chunk_overlap)
            doc.n_chunks = len(doc_chunks)
            chunks.extend(doc_chunks)
        self._chunks = chunks
        self._chunk_terms = [frozenset(tokenize(c.text)) for c in chunks]
        self._indexes = {}

    def _index(self, name: str) -> Retriever:
        with self._lock:
            if name not in self._indexes:
                idx = build_retriever(name)
                try:
                    idx.fit(self._chunks)
                except ImportError as exc:
                    raise RetrieverUnavailable(f"The '{name}' retriever needs an optional package: {exc}") from exc
                except Exception as exc:
                    if name in ("dense", "hybrid_dense"):  # model download / load problems
                        raise RetrieverUnavailable(
                            f"The '{name}' retriever could not load its embedding model "
                            "(it needs internet access the first time). Try another retriever."
                        ) from exc
                    raise
                self._indexes[name] = idx
            return self._indexes[name]

    # -------------------------------------------------------------- documents

    def add_document(self, filename: str, data: bytes) -> tuple[DocRecord, bool]:
        """Ingest a file. Returns (record, created); identical content is stored once."""
        name = safe_name(filename)
        digest = hashlib.sha256(data).hexdigest()
        doc_id = digest[:12]
        with self._lock:
            if doc_id in self._docs:
                return self._docs[doc_id], False
            pages = load_document(name, data, max_pages=self.settings.max_pages)
            record = DocRecord(doc_id, name, digest, time.time(), pages)
            self._docs[doc_id] = record
            self._rebuild()
            self._save_store()
            return record, True

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._docs:
                return False
            del self._docs[doc_id]
            self._rebuild()
            self._save_store()
            return True

    def list_documents(self) -> list[DocRecord]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.added_at)

    def get_document(self, doc_id: str) -> DocRecord | None:
        return self._docs.get(doc_id)

    # -------------------------------------------------------------- retrieval

    def retrieve(
        self,
        question: str,
        k: int | None = None,
        retriever: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[Hit]:
        k = k or self.settings.top_k
        with self._lock:
            if not self._chunks:
                return []
            index = self._index(retriever or self.settings.retriever)
            scores = index.scores(question)
            mask = None
            if doc_ids:
                wanted = set(doc_ids)
                mask = np.array([c.doc_id in wanted for c in self._chunks])
            return [Hit(self._chunks[i], s) for i, s in rank_top_k(scores, k, mask)]

    def _coverage(self, question: str, hits: list[Hit]) -> float:
        """Best share of the question's content words found in any one top chunk."""
        q_terms = set(tokenize(question))
        if not q_terms or not hits:
            return 0.0
        by_chunk = {id(c): t for c, t in zip(self._chunks, self._chunk_terms)}
        best = 0.0
        for hit in hits[:3]:
            terms = by_chunk.get(id(hit.chunk), frozenset())
            best = max(best, len(q_terms & terms) / len(q_terms))
        return best

    # ----------------------------------------------------------------- answer

    def _answerer(self) -> Answerer:
        if self._custom_answerer is not None:
            return self._custom_answerer
        bm25 = self._index("bm25")
        idf = bm25.idf if isinstance(bm25, BM25Retriever) else None
        return get_answerer(self.settings, idf)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        retriever: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        name = retriever or self.settings.retriever
        hits = self.retrieve(question, top_k, name, doc_ids)
        coverage = self._coverage(question, hits)

        def done(answer: str, abstained: bool, cites=(), answerer="none", warning=None) -> AnswerResult:
            ms = (time.perf_counter() - started) * 1000
            return AnswerResult(answer, abstained, list(cites), name, answerer, round(ms, 1), round(coverage, 2), warning)

        if not self._chunks:
            return done("No documents have been uploaded yet.", True)
        if not hits or coverage < self.settings.min_coverage:
            return done(NOT_FOUND, True)

        citations = [
            Citation(i, h.chunk.doc_id, h.chunk.doc_name, h.chunk.page, round(h.score, 4), h.chunk.text)
            for i, h in enumerate(hits, start=1)
        ]
        contexts = [Context(c.n, c.doc_name, c.page, c.text) for c in citations]

        answerer = self._answerer()
        warning = None
        try:
            text = answerer.answer(question, contexts)
        except LLMError as exc:
            log.warning("LLM call failed, using extractive fallback: %s", exc)
            warning = "The language model was unavailable, so this is an extractive answer."
            idf = self._index("bm25").idf  # type: ignore[attr-defined]
            answerer = ExtractiveAnswerer(idf, rank_decay=self.settings.rank_decay)
            text = answerer.answer(question, contexts)

        abstained = text.strip() == NOT_FOUND
        return done(text, abstained, [] if abstained else citations, answerer.name, warning)
