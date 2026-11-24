# DocChat

Ask questions about your own documents and get answers that cite the exact page they came from.

DocChat is a small retrieval-augmented generation (RAG) app: upload PDFs, text or Markdown files, ask a question, and read an answer with numbered citations. Click a citation to open the passage it came from, with the matching words highlighted. It runs fully offline out of the box and switches to Claude for written answers when you set an API key.

What makes it more than a demo is the evaluation harness (`eval/`). Every retrieval choice below was measured against a question set, selected on one half of the questions, and reported on the other half.

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000 and click "Load the sample documents"
```

For written answers from Claude instead of extracted sentences:

```bash
export ANTHROPIC_API_KEY=...        # optional
```

With Docker:

```bash
docker build -t docchat .
docker run -p 8000:8000 -v docchat-data:/data -e ANTHROPIC_API_KEY docchat
```

Run the tests and the evaluation:

```bash
pip install -r requirements-dev.txt
python -m pytest                    # 71 tests
python -m eval.run_eval             # rewrites eval/results.md and eval/results.json
python scripts/make_sample_docs.py  # regenerate the sample PDFs
```

## How it works

```
upload ─► loader ─► chunking ─► retriever ─► coverage check ─► answerer ─► answer + citations
          (pages)   (per page)   (BM25 /      (refuse if the    (Claude, or
                                  LSA / hybrid) passage barely   offline
                                                covers the       extractive)
                                                question)
```

| Module | Job |
|---|---|
| `app/loader.py` | PDF, TXT and MD to pages. PDFs keep real page numbers; text files are cut into ~2,000-character sections that play the same role. Repairs hyphenation and detects headings. |
| `app/chunking.py` | Two strategies: fixed word windows, and whole sentences packed up to a size limit. A chunk never crosses a page, so a citation always names a real page. |
| `app/retrieval.py` | BM25, TF-IDF, LSA (truncated SVD) and a hybrid that fuses BM25 and LSA with reciprocal rank fusion. An optional embedding retriever turns on when `fastembed` is installed. All share one small interface, so adding another is two methods. |
| `app/llm.py` | `AnthropicAnswerer` (grounded prompt, numbered `<source>` blocks, told to say so when the answer isn't there) and `ExtractiveAnswerer` (offline, picks the best-matching sentences). If the model call fails, the app falls back to the extractive answer and says so in the response. |
| `app/pipeline.py` | Ingestion, de-duplication by content hash, atomic persistence, document filters, the abstention rule, and the fallback. |
| `app/main.py` | FastAPI: upload, list, delete, ask, health, and the single-page UI. |

## Evaluation

The sample corpus is five small fictional PDFs (20 pages): two company policy documents that deliberately cover the same topics with different numbers, plus Git, gardening and personal finance. Questions come in three kinds:

- **41 keyword questions** that reuse the document's own wording.
- **20 paraphrase questions** that share little vocabulary with the answer ("How many weeks off does a new mother get" for a page about parental leave).
- **6 unanswerable questions**, two of them adversarial (right company name, topic not in the documents).

Retrieval is scored at page level. Questions are split into dev and test halves in `eval/gold.json`; the configuration is chosen on dev and reported on test. Full tables, including all 20 configurations, are in [`eval/results.md`](eval/results.md).

**Selected configuration (chosen on dev): sentence chunks of about 100 words, hybrid retrieval.**

| Split | Questions | Gold page ranked first | Gold page in top 3 | MRR |
|---|---:|---:|---:|---:|
| dev | 31 | 87% | 94% | 0.913 |
| test | 30 | 77% | 93% | 0.848 |

By question type, over all questions:

| Question type | Gold page ranked first | Gold page in top 3 | MRR |
|---|---:|---:|---:|
| Keyword (41) | 95% | 100% | 0.972 |
| Paraphrase (20) | 55% | 80% | 0.695 (95% CI 0.54 to 0.84) |

What the numbers say, and what they don't:

- **Keyword questions are nearly saturated; paraphrases are where retrieval is weak.** Every configuration scores 0.94 to 0.99 MRR on keyword questions, so that set can't tell them apart. Paraphrase MRR ranges from 0.587 to 0.695.
- **Chunking matters more than the retriever.** Averaged over retrievers, sentence-aligned 100-word chunks reach 0.681 paraphrase MRR against 0.600 to 0.622 for the 50-word and fixed-window variants. Averaged over chunkers, the four retrievers sit within 0.017 of each other (0.623 to 0.640), which is inside the noise.
- **The confidence intervals are wide.** With 20 paraphrase questions, one question moves hit@1 by 5 points. The ranking of the top few configurations is not statistically established; the gap between sentence-100w and the 50-word chunkers is the most credible finding.
- **Whole-page chunks look strong here for a trivial reason**: every page is exactly one topic and the labels are per page. Don't read it as "use whole pages".
- **The test split is small (30 questions).** The drop from 87% to 77% first-place hits between dev and test is the expected size of selection effect plus noise.

End-to-end with the offline extractive answerer (no language model): the correct fact appears in the answer for 95% of keyword questions and 40% of paraphrase questions, and the gold page is cited for 85% of answerable questions. A written answer from Claude should do better on paraphrases because the retrieved passages, not the sentence picker, become the limit; that is not measured here (see Status).

### Abstention is the weakest part

The app refuses to answer when the best retrieved passage contains less than a set share of the question's content words (`RAG_MIN_COVERAGE`, default 0.2). The sweep in `eval/results.md` shows the tradeoff: at 0.2 almost no answerable question is wrongly refused, but only 4 of 6 unanswerable questions are refused. The two that still get answered are the adversarial ones, "What is Northwind's stock ticker symbol?" and "Does Fabrikam offer health insurance to contractors?". The company name matches, so lexical coverage can't tell that the topic is missing. Raising the threshold to 0.5 refuses all six but wrongly refuses about 40% of answerable questions.

In Claude mode the model is instructed to answer "I couldn't find that" when the sources don't contain the answer, which is the better guard for this case. Adding a model-based answerability check or a cross-encoder reranker is the natural next step.

## Design decisions

- **Page-level citations over chunk-level.** A reader verifies an answer by opening a page, so chunks never cross page boundaries and every citation is a real page number.
- **Offline first.** Retrieval is BM25 and LSA with no model download, so the project runs anywhere, tests are deterministic, and the evaluation reproduces exactly. An embedding retriever is an opt-in extra rather than a requirement.
- **Hybrid via reciprocal rank fusion.** RRF needs no score calibration between retrievers, so fusing lexical and latent-semantic rankings is a few lines.
- **A refusal path, not just an answer path.** Wrong answers with confident citations are worse than "not found", so the pipeline can decline before calling the model.
- **Untrusted documents.** Document text is data, never instructions: the prompt says so, closing `</source>` tags inside a document are neutralised, and the UI inserts all text through DOM text nodes, never `innerHTML`, so uploaded content cannot inject markup.
- **Honest persistence.** Extracted text is stored in one JSON file with atomic writes, and a corrupt store is moved aside instead of crashing startup. Indexes are rebuilt in memory from that text, which is fast at this scale.

## API

Interactive docs are served at `/api/docs`.

| Method and path | Purpose |
|---|---|
| `POST /api/documents` | Upload one or more files (multipart `files`). Returns a status per file: `added`, `duplicate` or `error` with a reason. |
| `GET /api/documents` | List documents with page and chunk counts. |
| `DELETE /api/documents/{id}` | Remove a document and its text. |
| `POST /api/documents/samples` | Load the bundled sample PDFs. |
| `POST /api/ask` | `{"question", "top_k", "retriever", "doc_ids"}` returns the answer, `abstained`, `citations` (document, page, passage text), `latency_ms` and any `warning`. |
| `GET /api/health` | Version, document count, active answerer, available retrievers. |

## Configuration

All settings are environment variables; see `.env.example`. The defaults for chunking, retriever, abstention threshold and rank decay are the values selected by the evaluation, and they are defined in one place (`app/config.py`) that the harness imports.

## Public demo mode and deployment

Set `RAG_DEMO_MODE=1` to run a public demo. The sample documents are loaded at startup, nothing is written to disk, and answers always use the offline answerer. By default, uploads and deletes are disabled. For a portfolio demonstration only, `RAG_PUBLIC_UPLOADS=1` enables visitor uploads while keeping deletion disabled. Uploaded documents are shared with all visitors, held only in memory, and disappear when the service restarts or redeploys. Do not upload private or sensitive files. Set conservative `RAG_MAX_UPLOAD_MB` and `RAG_MAX_PAGES` limits. This mode has no authentication or per-visitor separation, so it is not suitable for confidential or production use.

The repository includes `render.yaml` for [Render](https://render.com), which can build the Dockerfile straight from GitHub. When this was written, Render's free web services slept after 15 minutes without traffic and took about a minute to wake, included 750 instance hours a month, and had a temporary filesystem. That last point is why a public deployment should use demo mode. Check Render's current free-tier terms before relying on them.

1. Push the repository to GitHub.
2. In Render, create a Blueprint from the repository so it applies `render.yaml`.
3. When the deploy finishes, open the `onrender.com` address Render shows and check that `/api/health` reports `"demo_mode": true`.

## Status and limitations

What has been run:

- 71 automated tests pass (loader, chunking, retrievers, pipeline, LLM wrapper with a fake client, and the HTTP API including bad uploads, oversized files and filename sanitising).
- The full evaluation runs end to end and produces the tables above.
- The UI was exercised in headless Chrome: loading samples, asking, expanding sources, an abstained question and a phone-width layout.

What has **not** been run in this build, so treat it as untested:

- **Claude mode against the real API.** The wrapper is tested with a fake client only; no API key was available.
- **The embedding retriever with a real model.** The model could not be downloaded in the build environment, so dense results are not in the evaluation. The retriever logic is tested with a stub embedder, and the app returns a clear 503 if the model can't load.
- **The Dockerfile.** It was written but Docker was not available to build it.
- **The Render deployment.** `render.yaml` follows Render's published Blueprint format but has not been deployed from this build.

Known limitations:

- Scanned PDFs need OCR, which is not supported; the upload is rejected with a message saying so.
- The evaluation corpus is small and fictional. Results show relative behaviour of the configurations on this corpus; absolute numbers will differ on real documents.
- The offline answerer matches words, not meaning. Asked "How long do Fabrikam contractors have to wait to be paid?", it returns a sentence about unpaid leave, because "contractors" and "paid" overlap with it, while the correct sentence says "paid within 45 days". It can also add a second, less relevant sentence. A language model, a reranker or embeddings are the natural fixes; none of them were measured here.
- Indexes live in memory and are rebuilt on every upload, which is fine for tens of documents but not for thousands.
- There is no authentication or per-user separation: everyone who can reach the app shares one library.

## Layout

```
app/        loader, chunking, retrieval, llm, pipeline, FastAPI app, static UI
data/       five sample PDFs
eval/       gold questions, harness, generated results
scripts/    sample PDF generator
tests/      71 tests
```
