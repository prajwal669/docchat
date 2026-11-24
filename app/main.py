"""FastAPI application: JSON API under /api plus the single-page UI at /."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .loader import LoaderError
from .pipeline import Pipeline
from .retrieval import RETRIEVERS, RetrieverUnavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("docchat")

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
SAMPLE_DIR = ROOT.parent / "data" / "sample_docs"

def _load_samples(pipeline: Pipeline) -> int:
    added = 0
    for path in sorted(SAMPLE_DIR.glob("*.pdf")):
        _, created = pipeline.add_document(path.name, path.read_bytes())
        added += int(created)
    return added


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    retriever: str | None = Field(default=None, max_length=32)
    doc_ids: list[str] | None = Field(default=None, max_length=100)


def _doc_json(d) -> dict:
    return {
        "id": d.id,
        "name": d.name,
        "pages": len(d.pages),
        "chunks": d.n_chunks,
        "added_at": d.added_at,
    }


def create_app(pipeline: Pipeline | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="DocChat", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.pipeline = pipeline or Pipeline(settings, persist=not settings.demo_mode)
    if settings.demo_mode:
        _load_samples(app.state.pipeline)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    def pipe() -> Pipeline:
        return app.state.pipeline

    def forbid_in_demo() -> None:
        if settings.demo_mode:
            raise HTTPException(
                403, "This public demo is read-only. Run DocChat on your own computer to use your own documents."
            )

    @app.get("/api/health")
    def health():
        p = pipe()
        return {
            "status": "ok",
            "version": __version__,
            "documents": len(p.list_documents()),
            "answerer": p._answerer().name,
            "retrievers": sorted(RETRIEVERS),
            "default_retriever": settings.retriever,
            "max_upload_mb": settings.max_upload_mb,
            "demo_mode": settings.demo_mode,
        }

    @app.get("/api/documents")
    def list_documents():
        return [_doc_json(d) for d in pipe().list_documents()]

    @app.post("/api/documents")
    async def upload(files: list[UploadFile] = File(...)):
        forbid_in_demo()
        results = []
        for f in files:
            name = f.filename or "document"
            data = await f.read(max_bytes + 1)
            if len(data) > max_bytes:
                results.append({"name": name, "status": "error",
                                "error": f"File is larger than {settings.max_upload_mb} MB."})
                continue
            try:
                record, created = await run_in_threadpool(pipe().add_document, name, data)
            except LoaderError as exc:
                results.append({"name": name, "status": "error", "error": str(exc)})
                continue
            except Exception:  # never leak internals to the client
                log.exception("Unexpected failure ingesting %s", name)
                results.append({"name": name, "status": "error", "error": "Could not process this file."})
                continue
            results.append({"name": record.name, "status": "added" if created else "duplicate",
                            "document": _doc_json(record)})
        all_failed = all(r["status"] == "error" for r in results)
        return JSONResponse({"results": results}, status_code=422 if all_failed else 200)

    @app.post("/api/documents/samples")
    def load_samples():
        forbid_in_demo()
        if not any(SAMPLE_DIR.glob("*.pdf")):
            raise HTTPException(404, "Sample documents are not installed in this deployment.")
        added = _load_samples(pipe())
        return {"added": added, "documents": [_doc_json(d) for d in pipe().list_documents()]}

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str):
        forbid_in_demo()
        if not pipe().delete_document(doc_id):
            raise HTTPException(404, "Document not found.")
        return {"deleted": doc_id}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        known = {d.id for d in pipe().list_documents()}
        if req.doc_ids and not set(req.doc_ids) <= known:
            raise HTTPException(404, "One or more selected documents no longer exist.")
        if req.retriever and req.retriever not in RETRIEVERS:
            raise HTTPException(422, f"Unknown retriever '{req.retriever}'. Options: {sorted(RETRIEVERS)}")
        try:
            result = pipe().ask(req.question.strip(), req.top_k, req.retriever, req.doc_ids)
        except RetrieverUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        return asdict(result)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
