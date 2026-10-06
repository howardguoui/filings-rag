# CLAUDE.md: filings-rag

Question answering over SEC 10-K filings: hybrid retrieval in Postgres (pgvector + full-text, RRF), cross-encoder
rerank, cited answers, RAGAS evaluation, FastAPI demo deployed on Render.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[evals,dev]"
.venv/bin/ruff check . && .venv/bin/ruff format --check .
TEST_DATABASE_URL=postgresql://... .venv/bin/python -m pytest -q   # needs Postgres with pgvector
```

No Postgres? `docker compose up -d db` locally. In a cloud sandbox without Docker, install
`postgresql-16` + `postgresql-server-dev-16`, build pgvector from github.com/pgvector/pgvector, and run a
throwaway cluster on a local socket.

## Rules

- **Never fabricate results.** `evals/results/` and `benchmarks/results/` hold only real runs from
  `scripts/publish_results.sh` on the owner's machine. Cloud sessions usually can't reach SEC EDGAR, Hugging Face
  or the Anthropic API; tests use HashEmbedder, FakeLLM and fixture filings.
- The installed anthropic SDK rejects `temperature`; `tests/test_sdk_contract.py` guards the request shape.
- Every change keeps lint and tests green and adds tests for new behavior.
- One roadmap item per change; move it to "Shipped" in ROADMAP.md with the date when done.
- Commit to `main` with a descriptive message; never force-push.
