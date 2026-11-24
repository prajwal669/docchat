# Evaluation results

Corpus: 5 PDFs, 20 pages. Questions: 41 keyword + 20 paraphrase + 6 unanswerable. The dev/test split is frozen in `eval/gold.json`.

Retrieval is scored at page level (does the gold page appear among the first k distinct pages?). With this few questions, one question is worth 2-5 points, so differences of a few points are noise; the paraphrase MRR carries a 95% bootstrap interval.

Dense (embedding) retrievers were **not evaluated**: the embedding model could not be loaded (ValueError).

## Retrieval quality by configuration

| Chunker | Retriever | Chunks | Keyword hit@1 | Keyword MRR | Paraphrase hit@1 | Paraphrase hit@3 | Paraphrase MRR (95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| sentence-100w | hybrid | 35 | 95% | 0.972 | 55% | 80% | 0.695 (0.54-0.84) |
| sentence-100w | bm25 | 35 | 95% | 0.972 | 55% | 75% | 0.678 (0.50-0.83) |
| whole-page | bm25 | 20 | 98% | 0.988 | 50% | 75% | 0.644 (0.48-0.80) |
| sentence-100w | lsa | 35 | 95% | 0.972 | 50% | 80% | 0.668 (0.51-0.82) |
| whole-page | lsa | 20 | 95% | 0.976 | 50% | 75% | 0.645 (0.47-0.80) |
| whole-page | hybrid | 20 | 95% | 0.976 | 50% | 75% | 0.645 (0.48-0.80) |
| sentence-100w | tfidf | 35 | 93% | 0.955 | 55% | 80% | 0.683 (0.51-0.83) |
| whole-page | tfidf | 20 | 95% | 0.972 | 50% | 75% | 0.645 (0.47-0.80) |
| fixed-100w | bm25 | 35 | 95% | 0.972 | 50% | 75% | 0.636 (0.47-0.80) |
| fixed-100w | hybrid | 35 | 93% | 0.959 | 50% | 75% | 0.637 (0.47-0.80) |
| fixed-50w | hybrid | 59 | 95% | 0.972 | 45% | 75% | 0.602 (0.42-0.77) |
| fixed-50w | bm25 | 59 | 95% | 0.972 | 45% | 70% | 0.600 (0.42-0.77) |
| sentence-50w | bm25 | 84 | 93% | 0.959 | 50% | 70% | 0.622 (0.44-0.79) |
| fixed-50w | lsa | 59 | 95% | 0.972 | 45% | 75% | 0.595 (0.42-0.76) |
| fixed-50w | tfidf | 59 | 95% | 0.967 | 45% | 75% | 0.602 (0.42-0.77) |
| sentence-50w | tfidf | 84 | 93% | 0.955 | 50% | 70% | 0.622 (0.44-0.79) |
| sentence-50w | lsa | 84 | 90% | 0.947 | 50% | 70% | 0.622 (0.45-0.79) |
| sentence-50w | hybrid | 84 | 90% | 0.947 | 50% | 70% | 0.622 (0.45-0.79) |
| fixed-100w | lsa | 35 | 93% | 0.959 | 40% | 75% | 0.587 (0.42-0.74) |
| fixed-100w | tfidf | 35 | 90% | 0.939 | 45% | 70% | 0.608 (0.43-0.76) |

Sorted by combined MRR over both sets.

## Averages by component (mean MRR over every configuration that uses it)

| Component | Keyword MRR | Paraphrase MRR |
|---|---:|---:|
| retriever: bm25 | 0.972 | 0.636 |
| retriever: tfidf | 0.958 | 0.632 |
| retriever: lsa | 0.965 | 0.623 |
| retriever: hybrid | 0.965 | 0.640 |
| chunker: fixed-50w | 0.971 | 0.600 |
| chunker: fixed-100w | 0.957 | 0.617 |
| chunker: sentence-50w | 0.952 | 0.622 |
| chunker: sentence-100w | 0.967 | 0.681 |
| chunker: whole-page | 0.978 | 0.645 |

Caveat: every page in this corpus is exactly one topic and the gold labels are page-level, so whole-page chunks match the labels by construction. Treat the chunker comparison as a demonstration of the harness, not a general conclusion about chunk sizes.

## Selected configuration (chosen on dev only)

**sentence-100w + hybrid**

| Split | Questions | hit@1 | hit@3 | MRR |
|---|---:|---:|---:|---:|
| dev | 31 | 87% | 94% | 0.913 |
| test | 30 | 77% | 93% | 0.848 |
| combined | 61 | 82% | 93% | 0.881 |

## End-to-end with the extractive answerer (abstention threshold 0.2)

| Chunker | Retriever | Facts present (keyword) | Facts present (paraphrase) | Gold page cited | Wrongly refused | Correctly refused |
|---|---|---:|---:|---:|---:|---:|
| sentence-100w | bm25 | 95% | 40% | 85% | 2% | 67% |
| sentence-100w | hybrid | 95% | 40% | 85% | 2% | 67% |
| fixed-100w | bm25 | 95% | 40% | 84% | 2% | 67% |
| fixed-100w | hybrid | 95% | 40% | 84% | 2% | 67% |
| sentence-100w | lsa | 95% | 40% | 84% | 2% | 67% |
| sentence-100w | tfidf | 93% | 40% | 82% | 2% | 67% |
| whole-page | bm25 | 93% | 35% | 84% | 0% | 67% |
| whole-page | hybrid | 90% | 35% | 82% | 0% | 67% |

Top 8 configurations by answer accuracy; all of them are in `results.json`. The extractive answerer only copies sentences, so these numbers measure retrieval plus sentence selection, not the quality of a written answer from a language model.

## Abstention threshold sweep (selected configuration)

A question is refused when the best retrieved passage contains less than this share of the question's content words.

| Threshold | Wrongly refused (dev) | Correctly refused (dev) | Wrongly refused (test) | Correctly refused (test) |
|---:|---:|---:|---:|---:|
| 0.0 | 0% | 67% | 0% | 67% |
| 0.2 | 0% | 67% | 3% | 67% |
| 0.3 | 13% | 100% | 17% | 67% |
| 0.4 | 29% | 100% | 30% | 67% |
| 0.5 | 42% | 100% | 40% | 100% |
| 0.6 | 52% | 100% | 47% | 100% |

## Answer sentence selection: rank decay sweep (selected configuration)

The offline answerer divides each sentence's score by `1 + decay x (passage rank - 1)`, so a higher value trusts the retriever's ordering more. 'Facts present' is the share of questions whose answer contains the gold fact.

| Decay | Keyword facts (dev) | Paraphrase facts (dev) | Keyword facts (test) | Paraphrase facts (test) |
|---:|---:|---:|---:|---:|
| 0.0 | 86% | 30% | 95% | 40% |
| 0.3 | 90% | 50% | 95% | 30% |
| 0.6 | 95% | 50% | 95% | 30% |
| 1.0 | 90% | 50% | 95% | 20% |
| 2.0 | 90% | 50% | 95% | 20% |
| 4.0 | 90% | 50% | 95% | 20% |

## Where the selected configuration misses

| # | Set | Split | Question | Gold page | Gold rank | Top retrieved |
|---|---|---|---|---|---:|---|
| 26 | keyword | test | How do I tell git to ignore build files? | Git p4 | 3 | Git p2; Git p1; Git p4 |
| 33 | keyword | dev | What is a good way to stop slugs? | Home p3 | 2 | Git p1; Home p3; Home p1 |
| h1 | paraphrase | dev | How many weeks off does a new mother get at Northwind? | Northwind p2 | 2 | Northwind p1; Northwind p2; Git p3 |
| h2 | paraphrase | test | Can Northwind staff plug in a thumb drive? | Northwind p4 | 7 | Northwind p1; Git p2; Git p3 |
| h4 | paraphrase | test | Can I take a rental car on a Northwind trip without asking my boss? | Northwind p3 | 2 | Fabrikam p2; Northwind p3; Git p3 |
| h6 | paraphrase | test | How soon after a purchase must a Northwind employee file the paperwork to get their money back? | Northwind p3 | 8 | Git p3; Personal p4; Fabrikam p4 |
| h7 | paraphrase | dev | Is Fabrikam going to cover my flights and hotels? | Fabrikam p3 | 5 | Northwind p3; Home p4; Fabrikam p2 |
| h12 | paraphrase | test | How can I get a copy of someone else's project onto my machine? | Git p1 | 2 | Fabrikam p4; Git p1; Git p4 |
| h14 | paraphrase | test | What's a cheap way to keep greenfly off my plants? | Home p3 | 3 | Home p2; Home p4; Home p3 |
| h16 | paraphrase | test | How often should I flip my rotting kitchen scraps? | Home p1 | 2 | Home p2; Home p1; Git p2 |
| h19 | paraphrase | dev | How should I divide my paycheck between essentials and fun? | Personal p1 | 10 | Personal p2; Fabrikam p1; Home p1 |
