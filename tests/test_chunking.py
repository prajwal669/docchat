import pytest

from app.chunking import chunk_fixed, chunk_pages, chunk_sentence, split_sentences
from app.loader import Page


def page(text: str, number: int = 1) -> Page:
    return Page(number, text)


def test_fixed_windows_overlap_exactly():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_fixed(page(" ".join(words)), size=100, overlap=20)
    assert len(chunks) == 3
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    assert chunks[-1].split()[-1] == "w249"  # nothing is dropped


def test_fixed_short_page_is_one_chunk_and_empty_is_none():
    assert chunk_fixed(page("just a few words"), 100, 10) == ["just a few words"]
    assert chunk_fixed(page(""), 100, 10) == []


def test_fixed_overlap_not_smaller_than_size_cannot_loop_forever():
    chunks = chunk_fixed(page(" ".join(["x"] * 50)), size=10, overlap=50)
    assert 1 < len(chunks) < 60


def test_split_sentences_keeps_abbreviation_free_boundaries():
    s = split_sentences("First one. Second one! Third? Then 4 more. $5 is cheap.")
    assert s == ["First one.", "Second one!", "Third?", "Then 4 more.", "$5 is cheap."]


def test_sentence_chunks_never_split_a_sentence_and_overlap_by_one():
    sents = [f"This is sentence number {i} with a handful of words." for i in range(12)]
    chunks = chunk_sentence(page(" ".join(sents)), size=30, overlap=1)
    assert len(chunks) > 1
    for c in chunks:
        assert all(s in sents for s in split_sentences(c))
    first_last = split_sentences(chunks[0])[-1]
    assert split_sentences(chunks[1])[0] == first_last
    covered = {s for c in chunks for s in split_sentences(c)}
    assert covered == set(sents)


def test_oversized_sentence_still_becomes_a_chunk():
    long = " ".join(["word"] * 300) + "."
    assert chunk_sentence(page(long), size=50, overlap=1) == [long]


def test_chunk_pages_never_cross_pages_and_keep_metadata():
    pages = [page("Alpha beta gamma. " * 40, 1), page("Delta epsilon zeta. " * 40, 2)]
    chunks = chunk_pages("d1", "doc.pdf", pages, "sentence", size=30, overlap=1)
    assert {c.page for c in chunks} == {1, 2}
    assert all(("Alpha" in c.text) == (c.page == 1) for c in chunks)
    assert all(c.doc_id == "d1" and c.doc_name == "doc.pdf" for c in chunks)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages("d", "n", [page("x y z")], "semantic")
