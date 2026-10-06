"""Compare answer-model latency and throughput: Ollama vs vLLM vs Claude API.

Every backend gets the same real RAG prompts (retrieved from your indexed filings),
so the comparison reflects this app's workload, not a toy prompt.

    python -m benchmarks.llm_latency --providers ollama vllm anthropic --runs 10

Writes benchmarks/results/latest.md and a timestamped JSON next to it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from filings_rag.config import get_settings
from filings_rag.llm import make_llm
from filings_rag.rag import SYSTEM_PROMPT, build_prompt

QUESTIONS = [
    "How does JPMorgan Chase govern cybersecurity risk?",
    "What drove NVIDIA's Data Center revenue growth?",
    "What risks does Apple describe about single-source suppliers?",
    "How sensitive is Bank of America's net interest income to interest rates?",
    "How did AWS operating income change year over year?",
]
OUT = Path(__file__).parent / "results"


def build_prompts(k: int) -> list[str]:
    from filings_rag.cli import _retriever

    r = _retriever(get_settings(), rerank=False)
    return [build_prompt(q, r.search(q, mode="hybrid", k=k).hits) for q in QUESTIONS]


def bench(provider: str, prompts: list[str], runs: int, max_tokens: int) -> dict:
    llm = make_llm(get_settings(), provider)
    llm.generate(SYSTEM_PROMPT, prompts[0], 16)  # warm-up: load weights / open connections
    lat, tps, out_tokens = [], [], []
    for i in range(runs):
        g = llm.generate(SYSTEM_PROMPT, prompts[i % len(prompts)], max_tokens)
        lat.append(g.latency_ms)
        if g.output_tokens:
            out_tokens.append(g.output_tokens)
            tps.append(g.output_tokens / (g.latency_ms / 1000))
    return {
        "model": llm.name,
        "runs": runs,
        "median_latency_ms": statistics.median(lat),
        "p90_latency_ms": sorted(lat)[max(0, int(len(lat) * 0.9) - 1)],
        "median_output_tokens": statistics.median(out_tokens) if out_tokens else None,
        "median_tokens_per_s": statistics.median(tps) if tps else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["ollama", "vllm"])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()

    prompts = build_prompts(a.k)
    results = {}
    for p in a.providers:
        t0 = time.perf_counter()
        try:
            results[p] = bench(p, prompts, a.runs, a.max_tokens)
        except Exception as exc:
            results[p] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(p, json.dumps(results[p]), f"({time.perf_counter() - t0:.0f}s)")

    OUT.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    (OUT / f"{stamp}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        f"# Answer-model benchmark, {stamp} UTC",
        "",
        f"{a.runs} runs per backend over {len(prompts)} real RAG prompts "
        f"(top {a.k} chunks), max {a.max_tokens} tokens.",
        "",
        "| Backend | Model | Median latency | p90 latency | Tokens/s |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p, r in results.items():
        if "error" in r:
            lines.append(f"| {p} | error: {r['error']} | | | |")
        else:
            tps = f"{r['median_tokens_per_s']:.0f}" if r["median_tokens_per_s"] else "–"
            lines.append(
                f"| {p} | {r['model']} | {r['median_latency_ms']:.0f} ms | {r['p90_latency_ms']:.0f} ms | {tps} |"
            )
    (OUT / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
