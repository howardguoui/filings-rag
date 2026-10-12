"""Retrieve, prompt, answer, and attach the sources the answer actually cited."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .llm import LLM, Generation, StreamingLLM
from .retrieval import Filters, Hit, Retriever

SYSTEM_PROMPT = """You answer questions about SEC 10-K filings using only the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Cite every factual sentence with the source numbers in square brackets, e.g. [2] or [1][3].
- Quote figures exactly as written, with units and the fiscal year they refer to.
- If the sources do not contain the answer, say "The filings provided do not contain this information." \
and stop.
- Be concise: at most 6 sentences or a short list."""

NOT_FOUND = "do not contain this information"
_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    n: int
    ticker: str
    company: str
    fiscal_year: int
    section: str
    url: str
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    contexts: list[str]
    mode: str
    model: str
    abstained: bool
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int | None] = field(default_factory=dict)


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{i}] {h.source_label}\n{h.text}" for i, h in enumerate(hits, 1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def _abstain(question: str, mode: str, model: str, timings: dict[str, float]) -> Answer:
    return Answer(question, "The filings provided do not contain this information.", [], [], mode, model, True, timings)


def _finish(question: str, hits: list[Hit], gen: Generation, mode: str, timings: dict[str, float]) -> Answer:
    nums = cited_numbers(gen.text, len(hits))
    citations = [
        Citation(
            n=n,
            ticker=hits[n - 1].ticker,
            company=hits[n - 1].company,
            fiscal_year=hits[n - 1].fiscal_year,
            section=f"Item {hits[n - 1].section_item}. {hits[n - 1].section_title}",
            url=hits[n - 1].url,
            snippet=hits[n - 1].text[:400],
        )
        for n in nums
    ]
    return Answer(
        question=question,
        answer=gen.text,
        citations=citations,
        contexts=[h.text for h in hits],
        mode=mode,
        model=gen.model,
        abstained=NOT_FOUND in gen.text.lower(),
        timings_ms=timings,
        usage={"input_tokens": gen.input_tokens, "output_tokens": gen.output_tokens},
    )


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLM,
    mode: str = "hybrid_rerank",
    k: int = 6,
    filters: Filters | None = None,
    max_tokens: int = 700,
) -> Answer:
    retrieved = retriever.search(question, mode=mode, k=k, filters=filters)
    hits = retrieved.hits
    timings = dict(retrieved.timings_ms)
    if not hits:
        return _abstain(question, retrieved.mode, llm.name, timings)

    gen = llm.generate(SYSTEM_PROMPT, build_prompt(question, hits), max_tokens)
    timings["generate_ms"] = gen.latency_ms
    return _finish(question, hits, gen, retrieved.mode, timings)


def stream_answer(
    question: str,
    retriever: Retriever,
    llm: StreamingLLM,
    mode: str = "hybrid_rerank",
    k: int = 6,
    filters: Filters | None = None,
    max_tokens: int = 700,
) -> Iterator[str | Answer]:
    """Like answer_question, but yield answer text as the model writes it, then the finished Answer.

    The Answer's timings gain `first_token_ms`: from the start of retrieval to the first answer text,
    i.e. how long a reader waits before anything appears. It is not part of the latency total.
    """
    t0 = time.perf_counter()
    retrieved = retriever.search(question, mode=mode, k=k, filters=filters)
    hits = retrieved.hits
    timings = dict(retrieved.timings_ms)
    if not hits:
        ans = _abstain(question, retrieved.mode, llm.name, timings)
        yield ans.answer
        yield ans
        return

    t_gen = time.perf_counter()
    gen: Generation | None = None
    for piece in llm.stream(SYSTEM_PROMPT, build_prompt(question, hits), max_tokens):
        if isinstance(piece, Generation):
            gen = piece
        else:
            yield piece
    if gen is None:
        raise RuntimeError(f"{llm.name} stream ended without a final result")
    timings["generate_ms"] = gen.latency_ms
    if gen.first_token_ms is not None:
        timings["first_token_ms"] = round((t_gen - t0) * 1000 + gen.first_token_ms, 1)
    yield _finish(question, hits, gen, retrieved.mode, timings)
