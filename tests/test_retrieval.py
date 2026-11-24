import numpy as np
import pytest

from app.chunking import Chunk
from app.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LsaRetriever,
    TfidfRetriever,
    build_retriever,
    rank_top_k,
    stem,
    tokenize,
)

TEXTS = [
    "Git branches let you work on features in isolation and merge them later.",
    "Compost needs brown leaves and green kitchen scraps turned every week.",
    "An emergency fund should cover three to six months of living expenses.",
    "Use git revert to undo a shared commit without rewriting history.",
    "Mulch keeps soil moist and suppresses weeds around vegetable plants.",
]
CHUNKS = [Chunk("d", "doc.pdf", i + 1, 0, t) for i, t in enumerate(TEXTS)]


def test_stemming_unifies_inflections_and_tokenize_drops_stopwords():
    assert stem("merges") == stem("merged") == stem("merge")
    assert stem("committed") == stem("commit")
    assert tokenize("What is the capital of France?") == ["capital", stem("france")]
    assert stem("paid") == stem("pay") == stem("paying")
    assert tokenize("50/30/20 rule") == ["50", "30", "20", "rule"]


@pytest.mark.parametrize("name", ["bm25", "tfidf", "lsa", "hybrid"])
def test_every_retriever_finds_the_right_chunk_first(name):
    r = build_retriever(name)
    r.fit(CHUNKS)
    scores = r.scores("how do I undo a commit that was already shared?")
    assert int(np.argmax(scores)) == 3


@pytest.mark.parametrize("name", ["bm25", "tfidf", "lsa", "hybrid"])
def test_no_overlap_means_no_scores(name):
    r = build_retriever(name)
    r.fit(CHUNKS)
    assert not r.scores("zebra xylophone quartz").any()


def test_bm25_prefers_rarer_terms():
    r = BM25Retriever()
    r.fit(CHUNKS)
    assert r.idf("revert") > r.idf("git")  # 'git' appears in two chunks, 'revert' in one
    assert r.idf("nonexistent") == 0.0


def test_retrievers_tolerate_tiny_and_empty_corpora():
    for cls in (BM25Retriever, TfidfRetriever, LsaRetriever, HybridRetriever):
        r = cls()
        r.fit(CHUNKS[:2])  # too small for LSA's low-rank model: must fall back, not crash
        assert r.scores("git branches").shape == (2,)
        r2 = cls()
        r2.fit([])
        assert r2.scores("anything").shape == (0,)


def test_rank_top_k_applies_mask_and_drops_zero_scores():
    scores = np.array([0.0, 0.9, 0.5, 0.7])
    assert rank_top_k(scores, 3) == [(1, 0.9), (3, 0.7), (2, 0.5)]
    mask = np.array([True, False, True, True])
    assert [i for i, _ in rank_top_k(scores, 3, mask)] == [3, 2]


def test_hybrid_fuses_two_rankings():
    r = HybridRetriever()
    r.fit(CHUNKS)
    scores = r.scores("git commit")
    assert scores[3] > 0 and scores[0] > 0 and scores[1] == 0


def test_dense_retriever_logic_with_a_stub_embedder():
    vocab = sorted({w for t in TEXTS for w in tokenize(t)})

    def fake_embed(texts):  # bag-of-words stand-in for a real embedding model
        return np.array([[tokenize(t).count(w) for w in vocab] for t in texts], dtype=float)

    r = DenseRetriever(embed_fn=fake_embed)
    r.fit(CHUNKS)
    assert int(np.argmax(r.scores("mulch for vegetable soil"))) == 4
    empty = DenseRetriever(embed_fn=fake_embed)
    empty.fit([])
    assert empty.scores("x").shape == (0,)


def test_unknown_retriever_name_is_an_error():
    with pytest.raises(ValueError):
        build_retriever("magic")


def test_lsa_falls_back_to_tfidf_on_small_corpora_and_never_crashes_on_stopwords():
    r = LsaRetriever()
    r.fit(CHUNKS)  # 5 chunks < MIN_CHUNKS
    assert r._svd is None
    assert int(np.argmax(r.scores("undo a shared commit"))) == 3
    junk = [Chunk("d", "x.txt", 1, 0, "the and of to a")] * 12  # stopwords only
    for cls in (TfidfRetriever, LsaRetriever, HybridRetriever):
        m = cls()
        m.fit(junk)
        assert not m.scores("anything").any()
