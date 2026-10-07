# Filings RAG

Ask questions about SEC 10-K filings and get answers cited to the exact section they came from.
Hybrid retrieval in Postgres (pgvector + full-text, fused with Reciprocal Rank Fusion), cross-encoder
reranking, and an evaluation suite that measures retrieval and answer quality with RAGAS.

**Stack:** Python, FastAPI, PostgreSQL + pgvector, fastembed (ONNX),
Claude API / Ollama / vLLM, RAGAS, Docker, GitHub Actions

## How it works

```mermaid
flowchart LR
    A[SEC EDGAR<br/>10-K HTML] --> B[Clean text<br/>tables kept as rows]
    B --> C[Split into Items<br/>1A, 1C, 7, 7A ...]
    C --> D[Chunk ~320 words<br/>+ contextual header]
    D --> E[(Postgres<br/>pgvector HNSW<br/>+ tsvector GIN)]
    Q[Question] --> F[Vector search] & G[Keyword search]
    E --> F & G
    F --> H[RRF fusion]
    G --> H
    H --> I[Cross-encoder<br/>rerank]
    I --> J[LLM answer<br/>with citations]
```

- **Section-aware parsing.** 10-Ks list every Item twice (table of contents, then the real section). The parser keeps
  the longest occurrence of each Item, so "Item 1C. Cybersecurity" chunks contain the actual cybersecurity
  disclosure. Only short heading lines count, so "see Item 7" in a paragraph doesn't split a section. Filings built
  around a cross-reference index instead of Item headings (common for large banks) are detected and indexed whole.
- **Contextual chunk headers.** Each chunk is embedded with a line like
  `Bank of America (BAC) 10-K FY2025, Item 7A. Market Risk`, so "rates rose 100 bps" is tied to a company and year.
- **Hybrid search in one SQL query.** Vector similarity (HNSW, cosine) finds paraphrases; full-text search finds exact
  terms like "Basel III" or "value-at-risk". Keyword search matches any of the question's terms (an all-terms match
  returns nothing for most natural questions) and ranks chunks with more of them higher. Both rankings are fused
  with RRF: `score = Σ 1 / (60 + rank)`. Filtered vector search uses pgvector's iterative HNSW scan, so filtering to
  one company still returns k results.
- **Reranking.** The top 40 fused candidates are rescored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`) that reads
  the question and passage together.
- **Grounded answers.** The model may only use the numbered sources, must cite every sentence, and must say
  "The filings provided do not contain this information" when they don't.
- **Three answer backends.** Claude API for the public demo, Ollama for free local runs, vLLM for fast local serving.

## Evaluation

`evals/questions.yaml` holds 36 questions across 8 companies: 32 answerable, each tagged with the 10-K Item that
answers it, and 4 unanswerable ones that test refusals.

| What | Metric |
| --- | --- |
| Retrieval | Hit rate@6 (the right company and Item is retrieved) and MRR, for vector, keyword, hybrid and hybrid + rerank, next to the exact score of a random ranking under the same filters |
| Answers (RAGAS, LLM judge) | Faithfulness, answer relevancy, context precision (no reference needed) |
| Behavior | Citation rate, correct refusals on unanswerable questions, false refusals |

Each run writes `evals/results/latest.md` and `latest.json`; the demo's Evaluations tab shows the latest run.
Commit the results folder so the deployed demo shows them.

Filings indexed whole (cross-reference index) count as misses in the section-level retrieval metric, since there is
no Item label to check.

## Run it locally

```bash
cp .env.example .env              # set SEC_USER_AGENT and a model provider
docker compose up -d db           # Postgres 16 + pgvector
pip install -e ".[evals,dev]"

filings-rag ingest AAPL MSFT NVDA AMZN JPM BAC GS TSLA   # latest 10-K for each company
filings-rag ask "How does Bank of America govern cybersecurity risk?" --tickers BAC
filings-rag serve                 # http://localhost:8080
```

**Evaluate:**

```bash
filings-rag eval --skip-generation                    # retrieval only, no LLM cost
filings-rag eval --llm ollama --judge ollama          # fully local answers + RAGAS
filings-rag eval --llm anthropic --judge anthropic    # Claude answers and grades
scripts/publish_results.sh                            # run the evals and push results to GitHub
```

**Benchmark answer backends (Ollama vs vLLM vs Claude):**

```bash
docker compose --profile vllm up -d vllm              # NVIDIA GPU, 16 GB is enough for the 7B AWQ model
python -m benchmarks.llm_latency --providers ollama vllm anthropic --runs 10
```

## Free static demo (GitHub Pages)

`filings-rag demo-export --llm ollama` answers a fixed set of example questions for real (same retrieval and
prompt as the live app), then writes the demo page, those recorded answers and the latest evaluation into `docs/`.
With GitHub Pages serving `docs/` from `main`, the page runs with no server, database or API key: visitors pick a
recorded question instead of typing their own, and the page says when and with which model each answer was made.

## Deploy to Render

A live version (visitors type their own questions) needs a server and an LLM API key, which costs money.


1. Push this repo to GitHub, then in Render choose **New → Blueprint** and pick the repo. `render.yaml` creates a
   free Postgres database and the Docker web service.
2. Render asks for `ANTHROPIC_API_KEY` when it creates the Blueprint (or set it later in the service's
   **Environment** tab; until then the app starts and `/api/health` reports the missing key). Set a monthly spend
   limit in the Anthropic console; the app also caps questions per minute and per day.
3. Load filings from your machine into the Render database (copy its **External Database URL**):
   `DATABASE_URL=<external url> filings-rag ingest`

Free Render web services sleep after 15 minutes idle (about a minute to wake), and free Postgres expires after
30 days.

## Tests

```bash
docker compose up -d db
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/filings pytest
```

Parsing, chunking, EDGAR pagination and retries, hybrid SQL against real pgvector, the API (validation, rate limits,
connection pool, health check, failures), the eval harness, and the exact request shape sent to the Anthropic SDK
and the RAGAS judge are covered. CI runs them on pushes to `main` and on pull requests, with a pgvector service
container.

## Layout

```
ROADMAP.md         what's next
src/filings_rag/   edgar.py (download) · sections.py · chunking.py · embeddings.py · db.py
                   retrieval.py (vector / keyword / hybrid SQL) · rerank.py · llm.py · rag.py · api.py · cli.py
evals/             questions.yaml · run_evals.py · results/
benchmarks/        llm_latency.py
```

Data comes from SEC EDGAR under its fair-access rules (identified User-Agent, under 10 requests/second).
Answers are generated from filings text and are not investment advice.
