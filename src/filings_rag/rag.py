"""Retrieve, prompt, answer, and attach the sources the answer actually cited."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from .llm import LLM, Generation, StreamingLLM
from .retrieval import Filters, Hit, Retriever
from .tracing import tracer

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


def _answer_span_attrs(mode: str, k: int, filters: Filters | None) -> dict:
    attrs: dict = {"rag.mode": mode, "rag.k": k}
    if filters and filters.tickers:
        attrs["rag.tickers"] = [t.upper() for t in filters.tickers]
    if filters and filters.fiscal_year:
        attrs["rag.fiscal_year"] = filters.fiscal_year
    return attrs


def _record_answer(span: Span, ans: Answer, n_hits: int) -> None:
    span.set_attributes(
        {
            "rag.hits": n_hits,
            "rag.citations": len(ans.citations),
            "rag.abstained": ans.abstained,
            "rag.retrieval_mode": ans.mode,
        }
    )


def _generate_span_attrs(llm: LLM, max_tokens: int) -> dict:
    # OpenTelemetry GenAI semantic conventions (gen_ai.*)
    return {"gen_ai.operation.name": "chat", "gen_ai.request.model": llm.name, "gen_ai.request.max_tokens": max_tokens}


def _record_generation(span: Span, gen: Generation) -> None:
    span.set_attribute("gen_ai.response.model", gen.model)
    if gen.input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", gen.input_tokens)
    if gen.output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", gen.output_tokens)


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLM,
    mode: str = "hybrid_rerank",
    k: int = 6,
    filters: Filters | None = None,
    max_tokens: int = 700,
) -> Answer:
    with tracer.start_as_current_span("answer", attributes=_answer_span_attrs(mode, k, filters)) as span:
        retrieved = retriever.search(question, mode=mode, k=k, filters=filters)
        hits = retrieved.hits
        timings = dict(retrieved.timings_ms)
        if not hits:
            ans = _abstain(question, retrieved.mode, llm.name, timings)
        else:
            with tracer.start_as_current_span("generate", attributes=_generate_span_attrs(llm, max_tokens)) as g:
                gen = llm.generate(SYSTEM_PROMPT, build_prompt(question, hits), max_tokens)
                _record_generation(g, gen)
            timings["generate_ms"] = gen.latency_ms
            ans = _finish(question, hits, gen, retrieved.mode, timings)
        _record_answer(span, ans, len(hits))
        return ans


def _fail(span: Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))


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

    Spans are started and ended by hand rather than made current across a `yield`: a web server may
    resume the generator on another thread, where a context attached on the first thread can't be
    detached. Each span still ends if the reader goes away mid-answer (the generator is closed).
    """
    root = tracer.start_span("answer", attributes=_answer_span_attrs(mode, k, filters))
    ctx = trace.set_span_in_context(root)
    try:
        t0 = time.perf_counter()
        with trace.use_span(root, end_on_exit=False):  # no yield inside: retrieval's spans nest under root
            retrieved = retriever.search(question, mode=mode, k=k, filters=filters)
        hits = retrieved.hits
        timings = dict(retrieved.timings_ms)
        if not hits:
            ans = _abstain(question, retrieved.mode, llm.name, timings)
            _record_answer(root, ans, 0)
            yield ans.answer
            yield ans
            return

        t_gen = time.perf_counter()
        gen: Generation | None = None
        g = tracer.start_span("generate", context=ctx, attributes=_generate_span_attrs(llm, max_tokens))
        try:
            for piece in llm.stream(SYSTEM_PROMPT, build_prompt(question, hits), max_tokens):
                if isinstance(piece, Generation):
                    gen = piece
                else:
                    yield piece
            if gen is None:
                raise RuntimeError(f"{llm.name} stream ended without a final result")
            _record_generation(g, gen)
            if gen.first_token_ms is not None:
                g.add_event("first_token", {"gen_ai.first_token_ms": gen.first_token_ms})
        except Exception as exc:
            _fail(g, exc)
            raise
        finally:
            g.end()
        timings["generate_ms"] = gen.latency_ms
        if gen.first_token_ms is not None:
            timings["first_token_ms"] = round((t_gen - t0) * 1000 + gen.first_token_ms, 1)
        ans = _finish(question, hits, gen, retrieved.mode, timings)
        _record_answer(root, ans, len(hits))
        yield ans
    except Exception as exc:
        _fail(root, exc)
        raise
    finally:
        root.end()
