"""Retrievers.

All retrievers share one interface: ``fit(chunks)`` and ``scores(query)``, where
``scores`` returns one relevance score per chunk. Ranking, filtering by
document and fusion are handled generically, so adding a new retriever (for
example a dense embedding model) means implementing those two methods and
registering it in ``RETRIEVERS``.

Implemented without any model downloads so the whole app runs offline:

* ``bm25``   - classic lexical ranking (Okapi BM25).
* ``tfidf``  - TF-IDF cosine similarity over unigrams and bigrams.
* ``lsa``    - latent semantic analysis (truncated SVD of TF-IDF), which can
               match related words that never co-occur in the query.
* ``hybrid`` - reciprocal rank fusion of BM25 and LSA.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
from collections import Counter, defaultdict
from typing import Callable, Protocol, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .chunking import Chunk

# --------------------------------------------------------------------------
# Text analysis
# --------------------------------------------------------------------------

STOPWORDS = frozenset(
    """a about after all also an and any are as at be been before being both but by
    can could did do does doing done for from had has have having he her here him his
    how i if in into is it its just me more most my no nor not of off on once only or
    other our out over own per same she should so some such than that the their them
    then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your
    many much long often each every need get got let""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# Common irregular past forms that a suffix stripper cannot connect to their base.
IRREGULAR = {
    "paid": "pay", "bought": "buy", "sold": "sell", "spent": "spend", "sent": "send",
    "kept": "keep", "held": "hold", "wrote": "write", "written": "write",
}


def stem(word: str) -> str:
    """A tiny suffix stripper: good enough to match 'merge/merges/merged'."""
    word = IRREGULAR.get(word, word)
    if len(word) <= 3 or word.isdigit():
        return word
    for suffix, repl in (
        ("ies", "y"),
        ("sses", "ss"),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                continue
            word = word[: len(word) - len(suffix)] + repl
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsfz":
        word = word[:-1]
    return word


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) == 1 and not tok.isdigit():
            continue
        if not keep_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok))
    return out


# --------------------------------------------------------------------------
# Retrievers
# --------------------------------------------------------------------------


class Retriever(Protocol):
    name: str

    def fit(self, chunks: Sequence[Chunk]) -> None: ...

    def scores(self, query: str) -> np.ndarray: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._len = np.zeros(0)
        self._avg = 0.0
        self.n = 0

    def fit(self, chunks: Sequence[Chunk]) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for i, chunk in enumerate(chunks):
            toks = tokenize(chunk.text)
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                postings[term].append((i, tf))
        self.n = len(chunks)
        self._len = np.array(lengths, dtype=float)
        self._avg = float(self._len.mean()) if self.n else 0.0
        self._postings = dict(postings)
        self._idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self._postings.items()
        }

    def idf(self, term: str) -> float:
        return self._idf.get(term, 0.0)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        if not self.n:
            return out
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self._len[i] / (self._avg or 1.0))
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class TfidfRetriever:
    name = "tfidf"

    def __init__(self):
        self._vec: TfidfVectorizer | None = None
        self._matrix = None
        self.n = 0

    def fit(self, chunks: Sequence[Chunk]) -> None:
        self.n = len(chunks)
        self._vec, self._matrix = None, None
        if not chunks:
            return
        vec = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        try:
            self._matrix = vec.fit_transform([c.text for c in chunks])
        except ValueError:  # empty vocabulary: every chunk is only stopwords
            return
        self._vec = vec

    def scores(self, query: str) -> np.ndarray:
        if not self.n or self._vec is None:
            return np.zeros(self.n)
        q = self._vec.transform([query])
        return np.asarray((self._matrix @ q.T).todense()).ravel()


class LsaRetriever:
    name = "lsa"
    MIN_CHUNKS = 10  # below this a low-rank model is meaningless; use TF-IDF instead

    def __init__(self, components: int = 100):
        self.components = components
        self._vec: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._docs = None
        self._fallback = TfidfRetriever()
        self.n = 0

    def fit(self, chunks: Sequence[Chunk]) -> None:
        self.n = len(chunks)
        self._fallback.fit(chunks)
        self._svd = None
        if self.n < self.MIN_CHUNKS:
            return
        vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        try:
            matrix = vec.fit_transform([c.text for c in chunks])
        except ValueError:  # empty vocabulary
            return
        k = min(self.components, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if k < 2:
            return
        self._vec = vec
        self._svd = TruncatedSVD(n_components=k, random_state=0)
        self._docs = normalize(self._svd.fit_transform(matrix))

    def scores(self, query: str) -> np.ndarray:
        if self._svd is None or self._vec is None:
            return self._fallback.scores(query)
        q = self._vec.transform([query])
        if q.nnz == 0:
            return np.zeros(self.n)
        qv = normalize(self._svd.transform(q))
        return self._docs @ qv.ravel()


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank per chunk (ties broken by index); zero-score chunks get no rank."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.zeros(len(scores))
    for r, idx in enumerate(order, start=1):
        if scores[idx] > 0:
            ranks[idx] = r
    return ranks


class HybridRetriever:
    """Reciprocal rank fusion (Cormack et al., 2009) of lexical + latent-semantic."""

    name = "hybrid"

    def __init__(self, rrf_k: int = 60, members: Sequence[Retriever] | None = None):
        self.rrf_k = rrf_k
        self.members: list[Retriever] = list(members) if members else [BM25Retriever(), LsaRetriever()]
        self.n = 0

    def fit(self, chunks: Sequence[Chunk]) -> None:
        self.n = len(chunks)
        for m in self.members:
            m.fit(chunks)

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(self.n)
        for m in self.members:
            ranks = _ranks(m.scores(query))
            mask = ranks > 0
            fused[mask] += 1.0 / (self.rrf_k + ranks[mask])
        return fused


class DenseRetriever:
    """Sentence-embedding retriever (optional extra: ``pip install fastembed``).

    Matches meaning rather than words, which is what the paraphrase questions in
    the evaluation need. The model is downloaded on first use, so it requires
    internet access once; set RAG_DENSE_MODEL to choose a different fastembed
    model. ``embed_fn`` lets tests inject a stub instead of the real model.
    """

    name = "dense"

    def __init__(self, model_name: str | None = None, embed_fn: Callable[[list[str]], np.ndarray] | None = None):
        self.model_name = model_name or os.getenv("RAG_DENSE_MODEL", "BAAI/bge-small-en-v1.5")
        self._embed_fn = embed_fn
        self._docs = np.zeros((0, 1))
        self.n = 0

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._embed_fn is None:
            from fastembed import TextEmbedding  # optional dependency, imported lazily

            model = TextEmbedding(self.model_name)
            self._embed_fn = lambda xs: np.array(list(model.embed(xs)))
        return normalize(np.asarray(self._embed_fn(list(texts)), dtype=float))

    def fit(self, chunks: Sequence[Chunk]) -> None:
        self.n = len(chunks)
        self._docs = self._embed([c.text for c in chunks]) if chunks else np.zeros((0, 1))

    def scores(self, query: str) -> np.ndarray:
        if not self.n:
            return np.zeros(0)
        return self._docs @ self._embed([query])[0]


RETRIEVERS: dict[str, Callable[[], Retriever]] = {
    "bm25": BM25Retriever,
    "tfidf": TfidfRetriever,
    "lsa": LsaRetriever,
    "hybrid": HybridRetriever,
}
# Dense retrieval is opt-in: only offered when the optional dependency is installed.
if importlib.util.find_spec("fastembed") is not None:
    RETRIEVERS["dense"] = DenseRetriever
    RETRIEVERS["hybrid_dense"] = lambda: HybridRetriever(members=[BM25Retriever(), DenseRetriever()])


class RetrieverUnavailable(RuntimeError):
    """A retriever exists but could not be prepared (for example, its model failed to download)."""


def build_retriever(name: str) -> Retriever:
    if name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever '{name}'. Options: {sorted(RETRIEVERS)}")
    return RETRIEVERS[name]()


def rank_top_k(
    scores: np.ndarray, k: int, allowed: np.ndarray | None = None
) -> list[tuple[int, float]]:
    """Top-k (index, score) pairs with score > 0, optionally restricted by a boolean mask."""
    s = scores.copy()
    if allowed is not None:
        s[~allowed] = 0.0
    order = np.argsort(-s, kind="stable")[:k]
    return [(int(i), float(s[i])) for i in order if s[i] > 0]
