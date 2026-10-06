# Roadmap

Shipped items move to the bottom with the date. Evaluation numbers are only committed from real runs
(`scripts/publish_results.sh`); nothing in `evals/results/` or `benchmarks/results/` is hand-written.

## Next

- [ ] **First published evaluation:** ingest the 8 demo companies, run retrieval + RAGAS, commit
      `evals/results/latest.md`, and quote the headline numbers in the README.
- [ ] **Live demo on Render** with the URL in the README.
- [ ] **Streaming answers:** server-sent events from `/api/ask` so the page shows tokens as they arrive; measure
      time to first token alongside total latency.
- [ ] **Query rewriting for multi-company questions:** split "compare Apple and Microsoft on X" into one retrieval
      per company, then merge, and add comparison questions to `evals/questions.yaml`.
- [ ] **Reranker ablation:** report hit rate and latency with and without the cross-encoder, and with 20 vs 40
      candidates, so the reranking cost is justified by numbers.
- [ ] **Multi-year filings:** ingest 3 fiscal years per company and add year-over-year questions
      ("how did risk factors change from FY2024 to FY2025").
- [ ] **Table-aware chunks:** keep financial tables whole with their caption so numeric questions retrieve the
      table, plus numeric questions in the eval set.
- [ ] **Tracing:** OpenTelemetry spans for embed / search / rerank / generate, exported to the console or an
      OTLP endpoint, so slow requests can be broken down.
- [ ] **Answer cache:** cache answers by normalized question + filters in Postgres to cut repeat cost on the demo.

## Shipped

- 2026-10-06: hybrid pgvector + full-text retrieval with RRF and cross-encoder rerank, cited answers from Claude /
  Ollama / vLLM, RAGAS evaluation with an exact random baseline, FastAPI demo with rate limits, Render blueprint, CI.
