from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.pipeline import Pipeline

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(storage_dir=tmp_path / "store", **overrides)


@pytest.fixture(scope="session")
def sample_files() -> dict[str, bytes]:
    files = {p.name: p.read_bytes() for p in sorted(SAMPLE_DIR.glob("*.pdf"))}
    assert len(files) == 5, "run `python scripts/make_sample_docs.py` first"
    return files


@pytest.fixture()
def pipeline(tmp_path, sample_files) -> Pipeline:
    pipe = Pipeline(make_settings(tmp_path), persist=False)
    for name, data in sample_files.items():
        pipe.add_document(name, data)
    return pipe


@pytest.fixture()
def client(tmp_path) -> TestClient:
    settings = make_settings(tmp_path)
    return TestClient(create_app(Pipeline(settings, persist=True), settings))
