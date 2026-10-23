# Roadmap

Shipped items move to the bottom with the date. Evaluation numbers are only committed from real runs
(`scripts/publish_results.sh`); nothing in `evals/results/` or `benchmarks/results/` is hand-written.

## Next

- [ ] **Live demo on Render** (visitors type their own questions) once an API budget is worth it; the free static
      demo covers the portfolio use for now.
- [ ] **Reranker ablation:** report hit rate and latency with and without the cross-encoder, and with 20 vs 40
      candidates, so the reranking cost is justified by numbers.
- [ ] **Multi-year filings:** ingest 3 fiscal years per company and add year-over-year questions
      ("how did risk factors change from FY2024 to FY2025").
- [ ] **Tracing:** OpenTelemetry spans for embed / search / rerank / generate, exported to the console or an
      OTLP endpoint, so slow requests can be broken down.
- [ ] **Answer cache:** cache answers by normalized question + filters in Postgres to cut repeat cost on the demo.

## Shipped

- 2026-10-23: table-aware chunks: financial tables become chunks of their own with their caption (the lead-in
  sentence), split by rows with the caption and header row repeated when too long; layout tables stay prose.
  `filings-rag ingest --rechunk` rebuilds the chunks of filings already indexed. Six numeric questions in
  `evals/questions.yaml`, with their own hit rate and MRR in the report.
- 2026-10-16: multi-company questions: a question naming two or more indexed companies (ticker or name), or one
  asked with several companies selected and none named, is searched once per company and the rankings are
  interleaved, so every company gets sources. Four comparison questions in `evals/questions.yaml`, and the report
  shows how often every company is covered with and without the split.
- 2026-10-12: streaming answers: `POST /api/ask/stream` sends server-sent events (tokens, then the cited answer)
  from Claude, Ollama and vLLM, the demo page renders tokens as they arrive, and time to first token is reported in
  the answer timings and the `benchmarks.llm_latency` table.
- 2026-10-07: free static demo on GitHub Pages (howardguoui.github.io/filings-rag): `filings-rag demo-export`
  records 10 real answers (qwen3:8b, hybrid + rerank) and the latest evaluation into `docs/`.
- 2026-10-07: first published evaluation (RTX 5070 Ti, local qwen3:8b answers and judge): hybrid hit rate@6 84% /
  MRR 0.83 vs 63% / 0.39 random; faithfulness 97% (30/32 scored), answer relevancy 91%, context precision 89%.
- 2026-10-06: hybrid pgvector + full-text retrieval with RRF and cross-encoder rerank, cited answers from Claude /
  Ollama / vLLM, RAGAS evaluation with an exact random baseline, FastAPI demo with rate limits, Render blueprint, CI.
