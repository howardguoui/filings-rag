"""Evaluate retrieval and answers.

1. Retrieval (no LLM, deterministic): for each mode, hit rate@k and MRR against the
   expected 10-K section, plus median search latency.
2. Answers (RAGAS 0.4, LLM-as-judge): faithfulness, answer relevancy and context
   precision on answerable questions; citation rate; correct refusals on
   unanswerable questions.

Writes evals/results/<timestamp>.json, latest.json and latest.md.

    filings-rag eval                                   # all modes, answers by LLM_PROVIDER
    filings-rag eval --skip-generation                 # retrieval only, free
    filings-rag eval --llm ollama --judge ollama       # fully local
    filings-rag eval --llm anthropic --judge anthropic --limit 10
"""

from __future__ import annotations

import asyncio
import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from filings_rag.config import Settings, get_settings
from filings_rag.retrieval import Filters, Retriever

HERE = Path(__file__).parent
RESULTS = HERE / "results"


@dataclass
class Question:
    id: str
    question: str
    tickers: list[str]
    sections: list[str]
    filter_tickers: bool
    unanswerable: bool

    @property
    def filters(self) -> Filters:
        return Filters(tickers=self.tickers if self.filter_tickers else None)


def load_questions(path: Path = HERE / "questions.yaml") -> list[Question]:
    raw = yaml.safe_load(path.read_text())["questions"]
    return [
        Question(
            id=q["id"],
            question=q["question"],
            tickers=[t.upper() for t in q.get("tickers", [])],
            sections=[s.upper() for s in q.get("expect", {}).get("sections", [])],
            filter_tickers=q.get("filter_tickers", True),
            unanswerable=q.get("unanswerable", False),
        )
        for q in raw
    ]


def is_hit(q: Question, ticker: str, section: str) -> bool:
    return ticker.upper() in q.tickers and section.upper() in q.sections


def retrieval_metrics(retriever: Retriever, questions: list[Question], modes: list[str], k: int) -> dict:
    """Hit rate@k, MRR and median search latency per retrieval mode."""
    answerable = [q for q in questions if not q.unanswerable]
    out: dict[str, dict] = {}
    for mode in modes:
        if mode == "hybrid_rerank" and retriever.reranker is None:
            continue  # reranker not loaded in this environment
        ranks, times, per_q = [], [], []
        for q in answerable:
            res = retriever.search(q.question, mode=mode, k=k, filters=q.filters)
            rank = next((i for i, h in enumerate(res.hits, 1) if is_hit(q, h.ticker, h.section_item)), None)
            ranks.append(rank)
            times.append(sum(res.timings_ms.values()))
            per_q.append({"id": q.id, "rank": rank})
        out[mode] = {
            "hit_rate": sum(r is not None for r in ranks) / len(ranks),
            "mrr": sum(1 / r for r in ranks if r) / len(ranks),
            "median_ms": statistics.median(times),
            "per_question": per_q,
        }
    return out


class _RagasEmbeddings:
    """Adapter so RAGAS can use this project's embedder for answer relevancy."""

    def __new__(cls, embedder):
        from ragas.embeddings import BaseRagasEmbedding

        class Adapter(BaseRagasEmbedding):
            def embed_text(self, text: str, **kw) -> list[float]:
                return [float(x) for x in embedder.embed_query(text)]

            async def aembed_text(self, text: str, **kw) -> list[float]:
                return self.embed_text(text)

        return Adapter()


def make_judge(settings: Settings, provider: str):
    from ragas.llms import llm_factory

    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        return llm_factory(
            settings.anthropic_model,
            provider="anthropic",
            client=AsyncAnthropic(api_key=settings.anthropic_api_key),
            max_tokens=2048,
        )
    from openai import AsyncOpenAI

    base, model = (
        (settings.ollama_base_url.rstrip("/") + "/v1", settings.ollama_model)
        if provider == "ollama"
        else (settings.vllm_base_url, settings.vllm_model)
    )
    return llm_factory(model, provider="openai", client=AsyncOpenAI(base_url=base, api_key="not-needed"))


async def _score_answers(rows: list[dict], judge, embeddings) -> None:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithoutReference, Faithfulness

    faith = Faithfulness(llm=judge)
    relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)
    precision = ContextPrecisionWithoutReference(llm=judge)
    for r in rows:
        args = {"user_input": r["question"], "response": r["answer"]}
        for name, metric, extra in (
            ("faithfulness", faith, {"retrieved_contexts": r["contexts"]}),
            ("answer_relevancy", relevancy, {}),
            ("context_precision", precision, {"retrieved_contexts": r["contexts"]}),
        ):
            try:
                r[name] = float((await metric.ascore(**args, **extra)).value)
            except Exception as exc:  # one bad judge call shouldn't sink the run
                r[name] = None
                r.setdefault("errors", []).append(f"{name}: {type(exc).__name__}: {exc}"[:300])


def generation_metrics(
    retriever: Retriever,
    questions: list[Question],
    settings: Settings,
    llm_name: str | None,
    judge_name: str | None,
    mode: str,
    k: int,
) -> dict:
    from filings_rag.llm import make_llm
    from filings_rag.rag import answer_question

    llm = make_llm(settings, llm_name)
    rows = []
    for q in questions:
        ans = answer_question(
            q.question, retriever, llm, mode=mode, k=k, filters=q.filters, max_tokens=settings.max_answer_tokens
        )
        rows.append(
            {
                "id": q.id,
                "question": q.question,
                "answer": ans.answer,
                "contexts": ans.contexts,
                "cited": len(ans.citations) > 0,
                "abstained": ans.abstained,
                "unanswerable": q.unanswerable,
                "latency_ms": ans.timings_ms.get("generate_ms"),
            }
        )

    answerable = [r for r in rows if not r["unanswerable"]]
    unanswerable = [r for r in rows if r["unanswerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    judge_label = None
    if judge_name and answerable:
        judge = make_judge(settings, judge_name)
        judge_label = getattr(judge, "model", None) or judge_name
        asyncio.run(_score_answers(answerable, judge, _RagasEmbeddings(retriever.embedder)))

    def mean(key: str) -> float | None:
        vals = [r[key] for r in answerable if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "model": llm.name,
        "judge": f"{judge_name}:{judge_label}" if judge_name else None,
        "mode": mode,
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("answer_relevancy"),
        "context_precision": mean("context_precision"),
        "citation_rate": (sum(r["cited"] for r in answered) / len(answered)) if answered else None,
        "false_refusals": sum(r["abstained"] for r in answerable) / max(1, len(answerable)),
        "abstention_rate": sum(r["abstained"] for r in unanswerable) / max(1, len(unanswerable)),
        "median_answer_ms": statistics.median([r["latency_ms"] or 0 for r in rows]) if rows else None,
        "rows": [{k2: v for k2, v in r.items() if k2 != "contexts"} for r in rows],
    }


def write_report(result: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = result["run_at"].replace(":", "").replace("-", "")[:15]
    (RESULTS / f"{stamp}.json").write_text(json.dumps(result, indent=2))
    (RESULTS / "latest.json").write_text(json.dumps(result, indent=2))

    def pct(x):
        return "–" if x is None else f"{x:.0%}"

    lines = [
        f"# Evaluation, {result['run_at']}",
        "",
        f"{result['n_questions']} questions · top {result['k']} chunks · embeddings `{result['embed_model']}`",
        "",
        "## Retrieval",
        "",
        "| Method | Hit rate | MRR | Median search |",
        "| --- | --- | --- | --- |",
    ]
    for mode, m in result["retrieval"].items():
        lines.append(f"| {mode} | {pct(m['hit_rate'])} | {m['mrr']:.2f} | {m['median_ms']:.0f} ms |")
    g = result.get("generation")
    if g:
        judged = f"judged by {g['judge']}" if g["judge"] else "not judged: pass --judge for RAGAS scores"
        lines += ["", f"## Answers ({g['model']}, {judged})", "", "| Metric | Score |", "| --- | --- |"]
        for name in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "citation_rate",
            "abstention_rate",
            "false_refusals",
        ):
            lines.append(f"| {name.replace('_', ' ')} | {pct(g[name])} |")
    path = RESULTS / "latest.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(
    modes: list[str],
    llm: str | None,
    judge: str | None,
    limit: int | None,
    skip_generation: bool,
    retriever: Retriever | None = None,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    questions = load_questions()[: limit or None]
    if retriever is None:
        from filings_rag.cli import _retriever

        retriever = _retriever(settings, rerank="hybrid_rerank" in modes)
    k = settings.top_k

    result = {
        "run_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "n_questions": sum(not q.unanswerable for q in questions),
        "k": k,
        "embed_model": settings.embed_model if settings.embed_provider == "fastembed" else settings.embed_provider,
        "retrieval": retrieval_metrics(retriever, questions, modes, k),
    }
    if not skip_generation:
        best = max(result["retrieval"], key=lambda m: result["retrieval"][m]["mrr"])
        result["generation"] = generation_metrics(retriever, questions, settings, llm, judge, best, k)
    report = write_report(result)
    print(report.read_text())
    return result
