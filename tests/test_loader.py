import pytest

from app.loader import (
    LoaderError,
    clean_text,
    load_document,
    load_pdf,
    load_text,
    mark_headings,
    safe_name,
)


def test_clean_text_repairs_hyphenation_and_flattens_wrapped_lines():
    raw = "The quick brown fox jumps over the lazy do-\ng and keeps\nrunning.\n\n\n\nNew paragraph."
    out = clean_text(raw)
    assert "dog" in out and "do-\ng" not in out
    assert "keeps running." in out  # single newline is layout, not meaning
    assert "\n\n" in out and "\n\n\n" not in out  # paragraphs survive, runs collapse


def test_mark_headings_detects_short_lines_only():
    page = (
        "Leave Policy\n"
        "Full-time employees accrue twenty four days of paid leave per year and may carry\n"
        "over some of them into the next year if they ask their manager before the deadline.\n"
        "A second paragraph line that is also fairly long so the page median stays high enough.\n"
    )
    out = mark_headings(page).split("\n")
    assert out[0] == "Leave Policy."
    assert not out[1].endswith(".") and out[2].endswith(".")  # wrapped body lines untouched


def test_mark_headings_handles_markdown():
    out = mark_headings("## Install steps\nRun the installer and follow the prompts to finish.")
    assert out.startswith("Install steps.")


def test_safe_name_blocks_path_traversal_and_odd_characters():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("C:\\Users\\me\\report final.pdf") == "report final.pdf"
    assert safe_name("we<ir>d|name?.txt") == "we_ir_d_name_.txt"
    assert safe_name("") == "document"


def test_load_text_splits_into_bounded_sections():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(12))
    pages = load_text(text.encode())
    assert len(pages) > 1
    assert [p.number for p in pages] == list(range(1, len(pages) + 1))
    assert all(len(p.text) <= 2400 for p in pages)


def test_load_text_rejects_empty():
    with pytest.raises(LoaderError):
        load_text(b"   \n\n  ")


@pytest.mark.parametrize("name,data", [("notes.docx", b"x"), ("noext", b"x"), ("a.txt", b"")])
def test_load_document_rejects_bad_input(name, data):
    with pytest.raises(LoaderError):
        load_document(name, data)


def test_load_pdf_keeps_real_page_numbers(sample_files):
    pages = load_pdf(sample_files["Git_Essentials.pdf"])
    assert [p.number for p in pages] == [1, 2, 3, 4]
    assert pages[1].text.startswith("Branching and Merging.")


def test_corrupt_pdf_raises_loader_error():
    with pytest.raises(LoaderError):
        load_pdf(b"%PDF-1.4\nthis is not really a pdf")
