"""Turn an uploaded file into a list of pages of clean text.

PDFs keep their real page numbers so citations point at the exact page.
Plain-text and Markdown files have no pages, so they are split into
~2,000-character sections that play the same role.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
SECTION_CHARS = 2000


class LoaderError(ValueError):
    """Raised for files we cannot or will not read."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    text: str


def mark_headings(text: str) -> str:
    """End heading-like lines with a period so they become their own sentence.

    PDF extraction gives no structure, so a heading is detected as a line that is
    much shorter than the page's typical line and does not end in punctuation.
    (Wrapped body lines are near full width and paragraph endings carry a period,
    so neither is mistaken for a heading.) Markdown '#' headings are handled too.
    """
    lines = [ln.strip() for ln in text.split("\n")]
    lengths = sorted(len(ln) for ln in lines if ln)
    median = lengths[len(lengths) // 2] if lengths else 0
    out = []
    for ln in lines:
        md = re.match(r"^#{1,6}\s+(.*\S)\s*#*$", ln)
        if md:
            ln = md.group(1)
            is_heading = True
        else:
            is_heading = bool(median) and len(lengths) >= 3 and len(ln) < 0.6 * median
        if ln and is_heading and len(ln.split()) <= 10 and not re.search(r"[.!?:;,]$", ln):
            ln += "."
        out.append(ln)
    return "\n".join(out)


def clean_text(text: str) -> str:
    """Normalise whitespace and repair PDF hyphenation at line ends."""
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Single newlines inside a paragraph are layout, not meaning.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def safe_name(filename: str) -> str:
    """Strip any directory components and odd characters from an uploaded name."""
    name = PurePath((filename or "").replace("\\", "/")).name
    name = re.sub(r"[^\w.\- ]", "_", name).strip(" .")
    return name[:120] or "document"


def extension(filename: str) -> str:
    return PurePath(filename).suffix.lower()


def _split_sections(text: str, limit: int = SECTION_CHARS) -> list[str]:
    """Pack paragraphs into sections of at most ~`limit` characters."""
    sections: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > limit:
            sections.append(current)
            current = ""
        # A single oversized paragraph is hard-split so no section is unbounded.
        while len(para) > limit:
            cut = para.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            sections.append(para[:cut].strip())
            para = para[cut:].strip()
        current = f"{current}\n\n{para}" if current else para
    if current:
        sections.append(current)
    return sections


def load_pdf(data: bytes, max_pages: int = 500) -> list[Page]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise LoaderError("Encrypted PDFs are not supported.")
        if len(reader.pages) > max_pages:
            raise LoaderError(f"PDF has more than {max_pages} pages.")
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = clean_text(mark_headings(page.extract_text() or ""))
            if text:
                pages.append(Page(number=i, text=text))
    except LoaderError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError) as exc:
        raise LoaderError(f"Could not read PDF: {exc}") from exc
    if not pages:
        raise LoaderError(
            "No extractable text found. Scanned PDFs need OCR, which is not supported yet."
        )
    return pages


def load_text(data: bytes) -> list[Page]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    text = text.replace("\r\n", "\n").replace("\x00", " ")
    # Keep paragraph breaks (clean_text would flatten them), normalise spaces only.
    text = re.sub(r"[ \t]+", " ", text)
    sections = _split_sections(text)
    if not sections:
        raise LoaderError("The file is empty.")
    return [Page(number=i, text=clean_text(mark_headings(s))) for i, s in enumerate(sections, start=1)]


def load_document(filename: str, data: bytes, max_pages: int = 500) -> list[Page]:
    ext = extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoaderError(
            f"Unsupported file type '{ext or 'unknown'}'. Use PDF, TXT or MD."
        )
    if not data:
        raise LoaderError("The file is empty.")
    if ext == ".pdf":
        return load_pdf(data, max_pages=max_pages)
    return load_text(data)
