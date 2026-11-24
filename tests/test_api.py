import io

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.pipeline import Pipeline

from .conftest import make_settings


def upload(client, name, data, ctype="application/pdf"):
    return client.post("/api/documents", files=[("files", (name, io.BytesIO(data), ctype))])


def test_health_and_index(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["answerer"] == "extractive" and "bm25" in h["retrievers"]
    r = client.get("/")
    assert r.status_code == 200 and "DocChat" in r.text


def test_upload_ask_delete_roundtrip(client, sample_files):
    res = upload(client, "Git_Essentials.pdf", sample_files["Git_Essentials.pdf"])
    assert res.status_code == 200
    doc = res.json()["results"][0]
    assert doc["status"] == "added" and doc["document"]["pages"] == 4

    assert upload(client, "again.pdf", sample_files["Git_Essentials.pdf"]).json()["results"][0]["status"] == "duplicate"
    assert [d["name"] for d in client.get("/api/documents").json()] == ["Git_Essentials.pdf"]

    ans = client.post("/api/ask", json={"question": "How do I recover a lost commit?"}).json()
    assert not ans["abstained"] and ans["citations"][0]["page"] == 3 and "reflog" in ans["answer"]

    assert client.delete(f"/api/documents/{doc['document']['id']}").status_code == 200
    assert client.delete(f"/api/documents/{doc['document']['id']}").status_code == 404
    assert client.post("/api/ask", json={"question": "How do I recover a lost commit?"}).json()["abstained"]


def test_bad_uploads_are_rejected_with_reasons(client):
    r = upload(client, "malware.exe", b"MZ\x90\x00")
    assert r.status_code == 422 and "Unsupported file type" in r.json()["results"][0]["error"]
    r = upload(client, "broken.pdf", b"%PDF-1.4 garbage")
    assert r.status_code == 422 and "Could not read PDF" in r.json()["results"][0]["error"]
    r = upload(client, "empty.txt", b"", "text/plain")
    assert r.status_code == 422


def test_mixed_batch_reports_each_file(client, sample_files):
    files = [("files", ("good.txt", io.BytesIO(b"Backups run nightly at 02:00 UTC."), "text/plain")),
             ("files", ("bad.exe", io.BytesIO(b"MZ"), "application/octet-stream"))]
    r = client.post("/api/documents", files=files)
    assert r.status_code == 200
    assert [x["status"] for x in r.json()["results"]] == ["added", "error"]


def test_oversized_upload_is_refused(tmp_path):
    settings = make_settings(tmp_path, max_upload_mb=1)
    c = TestClient(create_app(Pipeline(settings, persist=False), settings))
    r = c.post("/api/documents", files=[("files", ("big.txt", io.BytesIO(b"a " * 700_000), "text/plain"))])
    assert r.status_code == 422 and "larger than 1 MB" in r.json()["results"][0]["error"]


def test_uploaded_filename_is_sanitised(client):
    r = upload(client, "../../evil/../name.txt", b"Some harmless text about nothing in particular.", "text/plain")
    assert r.json()["results"][0]["name"] == "name.txt"


def test_ask_validation_and_errors(client, sample_files):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/api/ask", json={"question": "valid question", "top_k": 99}).status_code == 422
    r = client.post("/api/ask", json={"question": "valid question", "retriever": "magic"})
    assert r.status_code == 422 and "Unknown retriever" in r.json()["detail"]
    r = client.post("/api/ask", json={"question": "valid question", "doc_ids": ["nope"]})
    assert r.status_code == 404


def test_samples_endpoint_loads_the_demo_corpus(client):
    r = client.post("/api/documents/samples").json()
    assert r["added"] == 5 and len(r["documents"]) == 5
    assert client.post("/api/documents/samples").json()["added"] == 0  # idempotent
    ans = client.post("/api/ask", json={"question": "How long does Fabrikam take to pay contractor invoices?"}).json()
    assert "45 days" in ans["answer"]


def test_document_filter_over_http(client):
    docs = client.post("/api/documents/samples").json()["documents"]
    fab = next(d for d in docs if d["name"].startswith("Fabrikam"))
    ans = client.post("/api/ask", json={"question": "How many days of paid annual leave?", "doc_ids": [fab["id"]]}).json()
    assert all(c["doc_id"] == fab["id"] for c in ans["citations"])


def test_openapi_docs_available(client):
    assert client.get("/api/openapi.json").status_code == 200
