import io

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

from .conftest import make_settings


@pytest.fixture()
def demo(tmp_path):
    settings = make_settings(tmp_path, demo_mode=True)
    return TestClient(create_app(settings=settings)), settings


def test_demo_preloads_samples_and_is_read_only(demo):
    client, settings = demo
    health = client.get("/api/health").json()
    assert health["demo_mode"] is True and health["documents"] == 5
    docs = client.get("/api/documents").json()
    assert len(docs) == 5

    upload = client.post("/api/documents", files=[("files", ("x.txt", io.BytesIO(b"hello world text"), "text/plain"))])
    assert upload.status_code == 403 and "read-only" in upload.json()["detail"]
    assert client.delete(f"/api/documents/{docs[0]['id']}").status_code == 403
    assert client.post("/api/documents/samples").status_code == 403
    assert len(client.get("/api/documents").json()) == 5
    assert not (settings.storage_dir / "store.json").exists()  # a read-only demo writes nothing to disk


def test_demo_still_answers_questions(demo):
    client, _ = demo
    ans = client.post("/api/ask", json={"question": "How long does Fabrikam take to pay contractor invoices?"}).json()
    assert "45 days" in ans["answer"] and ans["citations"][0]["page"] == 3


def test_demo_never_uses_the_paid_model_even_with_a_key(demo, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    client, _ = demo
    assert client.get("/api/health").json()["answerer"] == "extractive"


def test_normal_mode_is_not_demo(client):
    assert client.get("/api/health").json()["demo_mode"] is False


def test_public_uploads_allowed_but_deletes_stay_disabled(tmp_path, sample_files):
    settings = make_settings(tmp_path, demo_mode=True, public_uploads=True, max_upload_mb=5, max_pages=50)
    client = TestClient(create_app(settings=settings))
    before = len(client.get("/api/documents").json())
    response = client.post(
        "/api/documents",
        files=[("files", ("visitor.txt", io.BytesIO(b"A public test document with enough text."), "text/plain"))],
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "added"
    docs = client.get("/api/documents").json()
    assert len(docs) == before + 1
    assert client.delete(f"/api/documents/{docs[-1]['id']}").status_code == 403
    health = client.get("/api/health").json()
    assert health["demo_mode"] is True and health["public_uploads"] is True
    assert not (settings.storage_dir / "store.json").exists()
