"""Runtime configuration, read from environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Chosen from the evaluation in eval/results.md (selected on the dev split).
DEFAULT_CHUNKER = "sentence"
DEFAULT_CHUNK_SIZE = 100
DEFAULT_RETRIEVER = "hybrid"
DEFAULT_MIN_COVERAGE = 0.2
DEFAULT_RANK_DECAY = 0.6  # how fast the offline answerer discounts sentences from lower-ranked passages


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Where the extracted text and metadata are persisted between restarts.
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_STORAGE_DIR", "storage")))

    # Chunking
    chunker: str = field(default_factory=lambda: os.getenv("RAG_CHUNKER", DEFAULT_CHUNKER))
    chunk_size: int = field(default_factory=lambda: _int("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE))  # words
    chunk_overlap: int = field(default_factory=lambda: _int("RAG_CHUNK_OVERLAP", 1))  # words or sentences

    # Retrieval
    retriever: str = field(default_factory=lambda: os.getenv("RAG_RETRIEVER", DEFAULT_RETRIEVER))
    top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 4))

    # Abstention: refuse to answer when the best chunks cover too little of the query.
    min_coverage: float = field(default_factory=lambda: _float("RAG_MIN_COVERAGE", DEFAULT_MIN_COVERAGE))

    rank_decay: float = field(default_factory=lambda: _float("RAG_RANK_DECAY", DEFAULT_RANK_DECAY))

    # Public demo: sample documents only, no uploads or deletes, offline answerer only.
    demo_mode: bool = field(
        default_factory=lambda: os.getenv("RAG_DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")
    )

    # Upload limits
    max_upload_mb: int = field(default_factory=lambda: _int("RAG_MAX_UPLOAD_MB", 20))
    max_pages: int = field(default_factory=lambda: _int("RAG_MAX_PAGES", 500))

    # Allow uploads in the public, in-memory demo. Uploaded files are shared by
    # all visitors and disappear when the service restarts or redeploys.
    public_uploads: bool = field(
        default_factory=lambda: os.getenv("RAG_PUBLIC_UPLOADS", "").strip().lower() in ("1", "true", "yes", "on")
    )

    # LLM (optional). Without ANTHROPIC_API_KEY the app uses the extractive answerer.
    llm_model: str = field(default_factory=lambda: os.getenv("RAG_LLM_MODEL", "claude-sonnet-5-5"))
    llm_max_tokens: int = field(default_factory=lambda: _int("RAG_LLM_MAX_TOKENS", 600))


def get_settings() -> Settings:
    return Settings()
