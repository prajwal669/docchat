"""Evaluate chunking strategies and retrievers on the sample corpus.

    python -m eval.run_eval

Protocol
--------
* Retrieval is scored at page level: a question is a hit at k if the gold
  (document, page) is among the first k distinct pages returned. Metrics:
  hit@1, hit@3, hit@5, MRR, and document-level hit@1.
* Two question sets: "keyword" questions reuse the document's wording, while
  "paraphrase" questions share little vocabulary with the answer. Keyword
  questions measure basic competence; paraphrase questions are where
  retrievers differ.
* Every (chunker x retriever) configuration is scored on everything, but the
  "selected" configuration is chosen on the dev split only and then reported on
  the held-out test split.
* End-to-end metrics use the offline extractive answerer: does the answer
  contain the gold facts, is the gold page cited, and does the system abstain
  on unanswerable questions.
* Dense (embedding) retrievers are included only when their model can be
  loaded; otherwise the report says they were skipped.

Writes eval/results.json and eval/results.md.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_MIN_COVERAGE, DEFAULT_RANK_DECAY, Settings  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.retrieval import RETRIEVERS as REGISTRY, DenseRetriever  # noqa: E402

SAMPLE_DIR = ROOT / "data" / "sample_docs"
GOLD_PATH = ROOT / "eval" / "gold.json"

CHUNKERS = [  # (label, strategy, size, overlap)
    ("fixed-50w", "fixed", 50, 10),
    ("fixed-100w", "fixed", 100, 20),
    ("sentence-50w", "sentence", 50, 1),
    ("sentence-100w", "sentence", 100, 1),
    ("whole-page", "fixed", 1_000_000, 0),
]
BASE_RETRIEVERS = ["bm25", "tfidf", "lsa", "hybrid"]
THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
DECAYS = [0.0, 0.3, 0.6, 1.0, 2.0, 4.0]
MAX_RANK = 10


def make_pipeline(strategy: str, size: int, overlap: int, retriever: str, min_coverage: float = DEFAULT_MIN_COVERAGE,
                  rank_decay: float = DEFAULT_RANK_DECAY) -> Pipeline:
    settings = Settings(
        storage_dir=Path("/nonexistent"), chunker=strategy, chunk_size=size, chunk_overlap=overlap,
        retriever=retriever, min_coverage=min_coverage, rank_decay=rank_decay, top_k=4,
    )
    pipe = Pipeline(settings, persist=False)
    for pdf in sorted(SAMPLE_DIR.glob("*.pdf")):
        pipe.add_document(pdf.name, pdf.read_bytes())
    return pipe


def page_ranks(pipe: Pipeline, question: str) -> list[tuple[str, int]]:
    """Distinct (doc, page) pairs in retrieval order, from the top MAX_RANK chunks."""
    seen: list[tuple[str, int]] = []
    for hit in pipe.retrieve(question, k=MAX_RANK):
        key = (hit.chunk.doc_name, hit.chunk.page)
        if key not in seen:
            seen.append(key)
    return seen


def first_rank(items: list, predicate) -> int | None:
    for i, item in enumerate(items, start=1):
        if predicate(item):
            return i
    return None


def retrieval_ranks(pipe: Pipeline, questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        pages = page_ranks(pipe, q["q"])
        rows.append({
            "id": q["id"], "split": q["split"], "set": q["set"],
            "rank": first_rank(pages, lambda p: p == (q["doc"], q["page"])),
            "doc_rank": first_rank(pages, lambda p: p[0] == q["doc"]),
            "top": pages[:3],
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ranks = [r["rank"] for r in rows]
    rr = np.array([1 / r if r else 0.0 for r in ranks])
    out = {"n": n, "mrr": float(rr.mean()) if n else 0.0}
    for k in (1, 3, 5):
        out[f"hit@{k}"] = sum(1 for r in ranks if r and r <= k) / n if n else 0.0
    out["doc@1"] = sum(1 for r in rows if r["doc_rank"] == 1) / n if n else 0.0
    return out


def bootstrap_ci(rows: list[dict], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    rr = np.array([1 / r["rank"] if r["rank"] else 0.0 for r in rows])
    rng = np.random.default_rng(seed)
    means = rng.choice(rr, size=(iters, len(rr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


CITE_RE = re.compile(r"\[(\d+)\]")


def end_to_end(pipe: Pipeline, answerable: list[dict], unanswerable: list[dict]) -> list[dict]:
    rows = []
    for q in answerable:
        res = pipe.ask(q["q"])
        text = res.answer.lower()
        cited = {int(n) for n in CITE_RE.findall(res.answer)}
        cited_pages = {(c.doc_name, c.page) for c in res.citations if c.n in cited}
        rows.append({
            "id": q["id"], "split": q["split"], "set": q["set"], "kind": "answerable",
            "abstained": res.abstained,
            "answer_ok": (not res.abstained) and all(s.lower() in text for s in q["answer_all"]),
            "cite_ok": (not res.abstained) and (q["doc"], q["page"]) in cited_pages,
        })
    for q in unanswerable:
        res = pipe.ask(q["q"])
        rows.append({"id": q["id"], "split": q["split"], "set": "unanswerable", "kind": "unanswerable",
                     "abstained": res.abstained})
    return rows


def e2e_summary(rows: list[dict]) -> dict:
    pct = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    ans = [r for r in rows if r["kind"] == "answerable"]
    una = [r for r in rows if r["kind"] == "unanswerable"]
    kw = [r for r in ans if r["set"] == "keyword"]
    pp = [r for r in ans if r["set"] == "paraphrase"]
    return {
        "answer_ok_keyword": pct([r["answer_ok"] for r in kw]),
        "answer_ok_paraphrase": pct([r["answer_ok"] for r in pp]),
        "cite_ok": pct([r["cite_ok"] for r in ans]),
        "false_abstain": pct([r["abstained"] for r in ans]),
        "correct_abstain": pct([r["abstained"] for r in una]),
    }


fmt = lambda x: f"{x * 100:.0f}%"


def dense_available() -> tuple[bool, str]:
    if "dense" not in REGISTRY:
        return False, "the optional `fastembed` package is not installed"
    try:
        DenseRetriever()._embed(["warm-up"])
        return True, ""
    except Exception as exc:  # download blocked, offline, etc.
        return False, f"the embedding model could not be loaded ({type(exc).__name__})"


def main() -> None:
    gold = json.loads(GOLD_PATH.read_text())
    keyword = [{**q, "set": "keyword"} for q in gold["questions"]]
    paraphrase = [{**q, "set": "paraphrase"} for q in gold["paraphrase"]]
    answerable = keyword + paraphrase
    unanswerable = gold["unanswerable"]
    by = lambda rows, key, val: [r for r in rows if r[key] == val]

    retrievers = list(BASE_RETRIEVERS)
    ok, why = dense_available()
    if ok:
        retrievers += ["dense", "hybrid_dense"]
        dense_note = "Dense (embedding) retrievers were evaluated."
    else:
        dense_note = f"Dense (embedding) retrievers were **not evaluated**: {why}."
    print(dense_note)

    results = []
    for label, strategy, size, overlap in CHUNKERS:
        for ret in retrievers:
            pipe = make_pipeline(strategy, size, overlap, ret)
            n_chunks = sum(d.n_chunks for d in pipe.list_documents())
            rr = retrieval_ranks(pipe, answerable)
            e2e = end_to_end(pipe, answerable, unanswerable)
            kw_rows, pp_rows = by(rr, "set", "keyword"), by(rr, "set", "paraphrase")
            results.append({
                "chunker": label, "retriever": ret, "chunks": n_chunks,
                "combined": summarize(rr), "keyword": summarize(kw_rows), "paraphrase": summarize(pp_rows),
                "dev": summarize(by(rr, "split", "dev")), "test": summarize(by(rr, "split", "test")),
                "paraphrase_mrr_ci95": bootstrap_ci(pp_rows),
                "e2e": e2e_summary(e2e),
                "rank_rows": rr,
            })
            r = results[-1]
            print(f"{label:14s} {ret:12s} chunks={n_chunks:3d} keyword MRR={r['keyword']['mrr']:.3f} "
                  f"paraphrase MRR={r['paraphrase']['mrr']:.3f}", flush=True)

    # ---- select on dev only (keyword + paraphrase questions together) ----
    selected = max(results, key=lambda r: (r["dev"]["mrr"], r["dev"]["hit@1"]))

    # ---- abstention threshold sweep for the selected configuration ----
    _, strategy, size, overlap = next(c for c in CHUNKERS if c[0] == selected["chunker"])
    sweep = []
    pipe = make_pipeline(strategy, size, overlap, selected["retriever"])
    for t in THRESHOLDS:
        pipe.settings = replace(pipe.settings, min_coverage=t)
        rows = end_to_end(pipe, answerable, unanswerable)
        sweep.append({"threshold": t, "dev": e2e_summary(by(rows, "split", "dev")),
                      "test": e2e_summary(by(rows, "split", "test"))})

    decay_sweep = []
    for d in DECAYS:
        rows = end_to_end(make_pipeline(strategy, size, overlap, selected["retriever"], rank_decay=d), answerable, unanswerable)
        decay_sweep.append({"rank_decay": d, "dev": e2e_summary(by(rows, "split", "dev")),
                            "test": e2e_summary(by(rows, "split", "test"))})

    gold_by_id = {q["id"]: q for q in answerable}
    misses = [r for r in selected["rank_rows"] if r["rank"] != 1]

    slim = [{k: v for k, v in r.items() if k != "rank_rows"} for r in results]
    (ROOT / "eval" / "results.json").write_text(json.dumps({
        "selected": {"chunker": selected["chunker"], "retriever": selected["retriever"]},
        "dense_evaluated": ok, "configs": slim, "abstention_sweep": sweep, "rank_decay_sweep": decay_sweep,
    }, indent=2))

    # ---- markdown report ----
    L: list[str] = []
    L.append("# Evaluation results\n")
    L.append(f"Corpus: {len(list(SAMPLE_DIR.glob('*.pdf')))} PDFs, 20 pages. Questions: {len(keyword)} keyword + "
             f"{len(paraphrase)} paraphrase + {len(unanswerable)} unanswerable. The dev/test split is frozen in "
             f"`eval/gold.json`.\n")
    L.append("Retrieval is scored at page level (does the gold page appear among the first k distinct pages?). "
             "With this few questions, one question is worth 2-5 points, so differences of a few points are noise; "
             "the paraphrase MRR carries a 95% bootstrap interval.\n")
    L.append(f"{dense_note}\n")
    L.append("## Retrieval quality by configuration\n")
    L.append("| Chunker | Retriever | Chunks | Keyword hit@1 | Keyword MRR | Paraphrase hit@1 | Paraphrase hit@3 | Paraphrase MRR (95% CI) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in sorted(results, key=lambda r: (-r["combined"]["mrr"], -r["combined"]["hit@1"])):
        k, p = r["keyword"], r["paraphrase"]; lo, hi = r["paraphrase_mrr_ci95"]
        L.append(f"| {r['chunker']} | {r['retriever']} | {r['chunks']} | {fmt(k['hit@1'])} | {k['mrr']:.3f} | "
                 f"{fmt(p['hit@1'])} | {fmt(p['hit@3'])} | {p['mrr']:.3f} ({lo:.2f}-{hi:.2f}) |")
    L.append("\nSorted by combined MRR over both sets.\n")

    L.append("## Averages by component (mean MRR over every configuration that uses it)\n")
    L.append("| Component | Keyword MRR | Paraphrase MRR |")
    L.append("|---|---:|---:|")
    for ret in retrievers:
        sel = [r for r in results if r["retriever"] == ret]
        L.append(f"| retriever: {ret} | {np.mean([r['keyword']['mrr'] for r in sel]):.3f} | "
                 f"{np.mean([r['paraphrase']['mrr'] for r in sel]):.3f} |")
    for c in CHUNKERS:
        sel = [r for r in results if r["chunker"] == c[0]]
        L.append(f"| chunker: {c[0]} | {np.mean([r['keyword']['mrr'] for r in sel]):.3f} | "
                 f"{np.mean([r['paraphrase']['mrr'] for r in sel]):.3f} |")
    L.append("\nCaveat: every page in this corpus is exactly one topic and the gold labels are page-level, "
             "so whole-page chunks match the labels by construction. Treat the chunker comparison as a demonstration "
             "of the harness, not a general conclusion about chunk sizes.\n")

    L.append("## Selected configuration (chosen on dev only)\n")
    s = selected
    L.append(f"**{s['chunker']} + {s['retriever']}**\n")
    L.append("| Split | Questions | hit@1 | hit@3 | MRR |")
    L.append("|---|---:|---:|---:|---:|")
    for sp in ("dev", "test", "combined"):
        v = s[sp]
        L.append(f"| {sp} | {v['n']} | {fmt(v['hit@1'])} | {fmt(v['hit@3'])} | {v['mrr']:.3f} |")

    L.append(f"\n## End-to-end with the extractive answerer (abstention threshold {DEFAULT_MIN_COVERAGE})\n")
    L.append("| Chunker | Retriever | Facts present (keyword) | Facts present (paraphrase) | Gold page cited | Wrongly refused | Correctly refused |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    top = sorted(results, key=lambda r: (-(r["e2e"]["answer_ok_keyword"] + r["e2e"]["answer_ok_paraphrase"]), -r["e2e"]["cite_ok"]))[:8]
    for r in top:
        e = r["e2e"]
        L.append(f"| {r['chunker']} | {r['retriever']} | {fmt(e['answer_ok_keyword'])} | {fmt(e['answer_ok_paraphrase'])} | "
                 f"{fmt(e['cite_ok'])} | {fmt(e['false_abstain'])} | {fmt(e['correct_abstain'])} |")
    L.append("\nTop 8 configurations by answer accuracy; all of them are in `results.json`. "
             "The extractive answerer only copies sentences, so these numbers measure retrieval plus sentence selection, "
             "not the quality of a written answer from a language model.\n")

    L.append("## Abstention threshold sweep (selected configuration)\n")
    L.append("A question is refused when the best retrieved passage contains less than this share of the question's content words.\n")
    L.append("| Threshold | Wrongly refused (dev) | Correctly refused (dev) | Wrongly refused (test) | Correctly refused (test) |")
    L.append("|---:|---:|---:|---:|---:|")
    for row in sweep:
        L.append(f"| {row['threshold']:.1f} | {fmt(row['dev']['false_abstain'])} | {fmt(row['dev']['correct_abstain'])} | "
                 f"{fmt(row['test']['false_abstain'])} | {fmt(row['test']['correct_abstain'])} |")

    L.append("\n## Answer sentence selection: rank decay sweep (selected configuration)\n")
    L.append("The offline answerer divides each sentence's score by `1 + decay x (passage rank - 1)`, so a higher value trusts the retriever's ordering more. "
             "'Facts present' is the share of questions whose answer contains the gold fact.\n")
    L.append("| Decay | Keyword facts (dev) | Paraphrase facts (dev) | Keyword facts (test) | Paraphrase facts (test) |")
    L.append("|---:|---:|---:|---:|---:|")
    for row in decay_sweep:
        L.append(f"| {row['rank_decay']:.1f} | {fmt(row['dev']['answer_ok_keyword'])} | {fmt(row['dev']['answer_ok_paraphrase'])} | "
                 f"{fmt(row['test']['answer_ok_keyword'])} | {fmt(row['test']['answer_ok_paraphrase'])} |")

    L.append("\n## Where the selected configuration misses\n")
    if misses:
        L.append("| # | Set | Split | Question | Gold page | Gold rank | Top retrieved |")
        L.append("|---|---|---|---|---|---:|---|")
        for m in misses:
            g = gold_by_id[m["id"]]
            tops = "; ".join(f"{d.split('_')[0]} p{p}" for d, p in m["top"])
            L.append(f"| {m['id']} | {m['set']} | {m['split']} | {g['q']} | {g['doc'].split('_')[0]} p{g['page']} | "
                     f"{m['rank'] or 'not found'} | {tops} |")
    else:
        L.append("None: the gold page was ranked first for every question.")
    (ROOT / "eval" / "results.md").write_text("\n".join(L) + "\n")

    print("\nSelected on dev:", selected["chunker"], selected["retriever"])
    for sp in ("dev", "test", "keyword", "paraphrase"):
        print(f"  {sp:10s}", {k: round(v, 3) for k, v in selected[sp].items()})
    print("wrote eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
