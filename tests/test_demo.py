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
