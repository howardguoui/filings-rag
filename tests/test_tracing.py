"""OpenTelemetry spans: one trace per question, with embed / search / rerank / generate under it."""

import logging

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from filings_rag import tracing
from filings_rag.api import create_app
from filings_rag.config import Settings
from filings_rag.llm import FakeLLM, Generation
from filings_rag.rag import answer_question, stream_answer
from filings_rag.rerank import KeywordOverlapReranker
from filings_rag.retrieval import Filters, Hit, RetrievalResult, Retriever

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def provider():
    # OpenTelemetry allows one global provider per process; install it once (if nothing else has),
    # keep it for the session, and read finished spans from the in-memory exporter.
    if isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        p = TracerProvider()
        p.add_span_processor(SimpleSpanProcessor(EXPORTER))
        trace.set_tracer_provider(p)
    else:
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(EXPORTER))


@pytest.fixture()
def spans():
    EXPORTER.clear()

    def finished() -> dict:
        return {s.name: s for s in EXPORTER.get_finished_spans()}

    return finished


def _hit(n: int) -> Hit:
    return Hit(n, "ACME", "Acme", 2025, "7", "MD&A", "hdr", f"Revenue rose 12% in fiscal 2025. Fact {n}.", "u", 1.0)


class Fixed:
    reranker = None

    def __init__(self, hits):
        self.hits = hits

    def search(self, *a, **k):
        return RetrievalResult(self.hits, "hybrid", {})


class Counted(FakeLLM):
    def generate(self, system, user, max_tokens):
        gen = super().generate(system, user, max_tokens)
        gen.input_tokens, gen.output_tokens = 120, 9
        return gen


def test_make_exporter_per_mode(monkeypatch):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    assert tracing.make_exporter("off") is None
    assert isinstance(tracing.make_exporter("console"), ConsoleSpanExporter)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    otlp = tracing.make_exporter("otlp")
    assert isinstance(otlp, OTLPSpanExporter)
    assert otlp._endpoint == "http://collector:4318/v1/traces"
    with pytest.raises(ValueError, match="TRACING must be one of"):
        tracing.make_exporter("jaeger")


def test_settings_reject_an_unknown_tracing_mode(monkeypatch):
    monkeypatch.setenv("TRACING", "loud")
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.setenv("TRACING", "otlp")
    assert Settings().tracing == "otlp"


def test_missing_sdk_gives_an_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_sdk(name, *a, **k):
        if name.startswith("opentelemetry.sdk"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_sdk)
    with pytest.raises(RuntimeError, match=r"pip install 'filings-rag\[tracing\]'"):
        tracing.make_exporter("console")


def test_configure_tracing_off_and_keeps_an_installed_provider(caplog):
    before = trace.get_tracer_provider()
    assert tracing.configure_tracing("off") is False
    with caplog.at_level(logging.INFO, logger="filings_rag.tracing"):
        assert tracing.configure_tracing("console") is False  # the test provider is already installed
    assert trace.get_tracer_provider() is before
    assert "already installed" in caplog.text


def test_answer_trace_has_generate_with_gen_ai_attributes(spans):
    answer_question("How fast did revenue grow?", Fixed([_hit(1), _hit(2)]), Counted(), k=2, max_tokens=300)
    s = spans()
    assert set(s) == {"answer", "generate"}
    root, gen = s["answer"], s["generate"]
    assert gen.parent.span_id == root.context.span_id and root.parent is None
    assert gen.attributes["gen_ai.request.model"] == "fake"
    assert gen.attributes["gen_ai.request.max_tokens"] == 300
    assert gen.attributes["gen_ai.usage.input_tokens"] == 120
    assert gen.attributes["gen_ai.usage.output_tokens"] == 9
    assert root.attributes["rag.hits"] == 2 and root.attributes["rag.citations"] == 1
    assert root.attributes["rag.abstained"] is False


def test_no_hits_traces_no_generate_span(spans):
    ans = answer_question("q", Fixed([]), FakeLLM())
    assert ans.abstained
    assert set(spans()) == {"answer"} and spans()["answer"].attributes["rag.abstained"] is True


def test_failed_generation_marks_both_spans_as_errors(spans):
    class Down:
        name = "down"

        def generate(self, *a):
            raise ConnectionError("model server unreachable")

    with pytest.raises(ConnectionError):
        answer_question("q", Fixed([_hit(1)]), Down())
    s = spans()
    assert s["generate"].status.status_code == StatusCode.ERROR
    assert s["answer"].status.status_code == StatusCode.ERROR
    assert s["generate"].events[0].name == "exception"


def test_retrieval_spans_nest_under_answer(db, embedder, spans):
    r = Retriever(db, embedder, KeywordOverlapReranker(), candidate_k=10)
    answer_question("How much did revenue grow?", r, FakeLLM(), mode="hybrid_rerank", k=3, filters=Filters(["ACME"]))
    s = spans()
    assert {"answer", "retrieve", "embed", "search", "rerank", "generate"} <= set(s)
    root = s["answer"].context.span_id
    assert s["retrieve"].parent.span_id == root and s["generate"].parent.span_id == root
    for child in ("embed", "search", "rerank"):
        assert s[child].parent.span_id == s["retrieve"].context.span_id
    assert s["search"].attributes["retrieval.tickers"] == ("ACME",)
    assert s["rerank"].attributes["rerank.candidates"] == s["search"].attributes["retrieval.rows"]
    assert s["retrieve"].attributes["retrieval.hits"] == 3


def test_split_question_has_one_search_span_per_company(db, embedder, spans):
    Retriever(db, embedder).search("Compare ACME and GLBX revenue", mode="hybrid", k=4)
    found = [x for x in EXPORTER.get_finished_spans() if x.name == "search"]
    assert sorted(x.attributes["retrieval.tickers"][0] for x in found) == ["ACME", "GLBX"]
    assert len([x for x in EXPORTER.get_finished_spans() if x.name == "embed"]) == 1  # embedded once
    assert spans()["retrieve"].attributes["retrieval.split_tickers"] == ("ACME", "GLBX")


def test_stream_trace_records_first_token(spans):
    out = list(stream_answer("How fast did revenue grow?", Fixed([_hit(1)]), FakeLLM()))
    assert isinstance(out[-1].timings_ms["first_token_ms"], float)
    s = spans()
    assert s["generate"].parent.span_id == s["answer"].context.span_id
    assert [e.name for e in s["generate"].events] == ["first_token"]
    assert s["answer"].attributes["rag.citations"] == 1


def test_stream_closed_early_still_ends_its_spans(spans):
    it = stream_answer("How fast did revenue grow?", Fixed([_hit(1)]), FakeLLM())
    next(it)  # first token, then the reader goes away
    it.close()
    s = spans()
    assert {"answer", "generate"} == set(s)
    assert s["answer"].status.status_code != StatusCode.ERROR


def test_stream_without_final_generation_is_an_error_span(spans):
    class Truncated(FakeLLM):
        def stream(self, *a):
            yield "partial"

    with pytest.raises(RuntimeError, match="without a final result"):
        list(stream_answer("q", Fixed([_hit(1)]), Truncated()))
    assert spans()["generate"].status.status_code == StatusCode.ERROR


def test_streaming_endpoint_trace_is_whole(db, embedder, spans, caplog):
    # The server resumes the event generator on worker threads; spans must still nest and detach cleanly.
    s = Settings(rerank_enabled=False, rate_limit_per_minute=10)
    app = create_app(s, conn=db, embedder=embedder, llm=FakeLLM(), reranker=KeywordOverlapReranker())
    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"), TestClient(app) as c:
        r = c.post("/api/ask/stream", json={"question": "What drove gross margin?", "tickers": ["ACME"]})
        assert r.status_code == 200 and "event: answer" in r.text
    assert "Failed to detach context" not in caplog.text
    got = spans()
    root = got["answer"].context.span_id
    assert got["retrieve"].parent.span_id == root and got["generate"].parent.span_id == root
    assert got["search"].parent.span_id == got["retrieve"].context.span_id
    assert {x.context.trace_id for x in EXPORTER.get_finished_spans()} == {got["answer"].context.trace_id}


def test_generation_record_skips_unknown_token_counts(spans):
    class NoUsage(FakeLLM):
        def generate(self, system, user, max_tokens):
            return Generation("Revenue rose [1].", "vllm:x", None, None, 1.0)

    answer_question("q", Fixed([_hit(1)]), NoUsage())
    attrs = spans()["generate"].attributes
    assert attrs["gen_ai.response.model"] == "vllm:x" and "gen_ai.usage.input_tokens" not in attrs


def test_cli_turns_tracing_on_from_settings(monkeypatch):
    from filings_rag import cli
    from filings_rag.config import get_settings

    modes = []
    monkeypatch.setattr(tracing, "configure_tracing", modes.append)
    monkeypatch.setattr(cli, "cmd_ask", lambda args: None)
    monkeypatch.setenv("TRACING", "console")
    get_settings.cache_clear()
    try:
        cli.main(["ask", "What drove gross margin?"])
    finally:
        get_settings.cache_clear()
    assert modes == ["console"]
