"""Command line: ingest filings, ask questions, run evaluations, serve the API.

filings-rag ingest AAPL MSFT NVDA JPM BAC --years 1
filings-rag ask "What cybersecurity risks does JPMorgan describe?" --tickers JPM
filings-rag eval --modes vector hybrid hybrid_rerank
filings-rag serve
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import get_settings

DEMO_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "BAC", "GS", "TSLA"]


def _retriever(settings, rerank: bool = True):
    from .db import connect
    from .embeddings import make_embedder
    from .retrieval import Retriever

    conn = connect(settings.database_url)
    reranker = None
    if rerank and settings.rerank_enabled:
        from .rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.rerank_model)
    return Retriever(conn, make_embedder(settings), reranker, settings.candidate_k)


def cmd_ingest(args) -> None:
    from .db import connect
    from .edgar import EdgarClient
    from .embeddings import make_embedder
    from .ingest import ingest_tickers

    s = get_settings()
    edgar = EdgarClient(s.sec_user_agent, f"{s.data_dir}/raw")
    counts = ingest_tickers(connect(s.database_url), make_embedder(s), edgar, args.tickers, args.years)
    for name, n in counts.items():
        print(f"{name}: {n} chunks")


def cmd_ask(args) -> None:
    from .llm import make_llm
    from .rag import answer_question
    from .retrieval import Filters

    s = get_settings()
    ans = answer_question(
        args.question,
        _retriever(s, rerank=args.mode == "hybrid_rerank"),
        make_llm(s, args.llm),
        mode=args.mode,
        k=args.k,
        filters=Filters(tickers=args.tickers, fiscal_year=args.year),
        max_tokens=s.max_answer_tokens,
    )
    print(ans.answer, "\n")
    for c in ans.citations:
        print(f"[{c.n}] {c.company} FY{c.fiscal_year}, {c.section}\n    {c.url}")
    print("\n" + json.dumps({"model": ans.model, "mode": ans.mode, "timings_ms": ans.timings_ms}))


def cmd_eval(args) -> None:
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))  # evals/ lives at the repo root; run this from there
    from evals.run_evals import main as run_evals

    run_evals(args.modes, args.llm, args.judge, args.limit, args.skip_generation)


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("filings_rag.api:app_factory", factory=True, host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="filings-rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="Download 10-Ks from EDGAR and index them")
    i.add_argument("tickers", nargs="*", default=DEMO_TICKERS)
    i.add_argument("--years", type=int, default=1, help="How many recent 10-Ks per company")
    i.set_defaults(func=cmd_ingest)

    a = sub.add_parser("ask", help="Ask a question from the command line")
    a.add_argument("question")
    a.add_argument("--tickers", nargs="*")
    a.add_argument("--year", type=int)
    a.add_argument("--mode", default="hybrid_rerank", choices=["vector", "keyword", "hybrid", "hybrid_rerank"])
    a.add_argument("--k", type=int, default=6)
    a.add_argument("--llm", choices=["anthropic", "ollama", "vllm", "fake"])
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="Run retrieval and RAGAS evaluations")
    e.add_argument("--modes", nargs="*", default=["vector", "keyword", "hybrid", "hybrid_rerank"])
    e.add_argument("--llm", choices=["anthropic", "ollama", "vllm", "fake"], help="Model that answers")
    e.add_argument("--judge", choices=["anthropic", "ollama", "vllm"], help="Model that grades (RAGAS)")
    e.add_argument("--limit", type=int, help="Only the first N questions")
    e.add_argument("--skip-generation", action="store_true", help="Retrieval metrics only (no LLM calls)")
    e.set_defaults(func=cmd_eval)

    sv = sub.add_parser("serve", help="Run the web API and demo page")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8080)
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
