"""Streamed answers: the LLM backends, stream_answer, and the /api/ask/stream server-sent events."""

import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from filings_rag.api import create_app
from filings_rag.config import Settings
from filings_rag.llm import FakeLLM, Generation, ThinkFilter
from filings_rag.rag import Answer, answer_question, stream_answer
from filings_rag.rerank import KeywordOverlapReranker
from filings_rag.retrieval import Filters, Hit, RetrievalResult, Retriever


def _hit(n: int) -> Hit:
    return Hit(n, "ACME", "Acme", 2025, "7", "MD&A", "hdr", f"Revenue rose 12% in fiscal 2025. Fact {n}.", "u", 1.0)


class Fixed:
    """A retriever that returns the given hits without a database."""

    reranker = None

    def __init__(self, hits):
        self.hits = hits

    def search(self, *a, **k):
        return RetrievalResult(self.hits, "hybrid", {"embed_ms": 1.0, "search_ms": 2.0})


def _feed(chunks: list[str]) -> str:
    f = ThinkFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


def test_think_filter_drops_a_leading_think_block_split_across_chunks():
    assert _feed(["<th", "ink>plan the ", "answer</th", "ink>\n\nRevenue rose [1]."]) == "Revenue rose [1]."
    assert _feed(["  ", "<think>", "x", "</think>", "A", " [1]."]) == "A [1]."


def test_think_filter_passes_ordinary_text_through_unchanged():
    assert _feed(["Revenue ", "rose <think> ", "[1]."]) == "Revenue rose <think> [1]."
    assert _feed(["<", "b>bold"]) == "<b>bold"
    assert _feed(["<thi"]) == "<thi"  # stream ended on something that only looked like a tag start


def test_think_filter_unclosed_block_yields_no_answer():
    assert _feed(["<think>still reasoning when tokens ran out"]) == ""


def test_fake_llm_stream_matches_generate():
    user = "Sources:\n\n[1] Acme\nRevenue rose 12%. More.\n\nQuestion: q"
    pieces = list(FakeLLM().stream("sys", user, 100))
    *text, final = pieces
    assert isinstance(final, Generation) and final.first_token_ms == 0.0
    assert "".join(text) == final.text == FakeLLM().generate("sys", user, 100).text


def test_stream_answer_yields_text_then_the_same_answer_as_answer_question():
    pieces = list(stream_answer("How did revenue change?", Fixed([_hit(1), _hit(2)]), FakeLLM()))
    *text, ans = pieces
    assert isinstance(ans, Answer) and all(isinstance(p, str) for p in text)
    assert "".join(text) == ans.answer
    plain = answer_question("How did revenue change?", Fixed([_hit(1), _hit(2)]), FakeLLM())
    assert (ans.answer, ans.citations, ans.abstained) == (plain.answer, plain.citations, plain.abstained)
    t = ans.timings_ms
    assert t["embed_ms"] == 1.0 and "generate_ms" in t
    assert t["first_token_ms"] >= 0  # measured from the start of retrieval


def test_stream_answer_without_hits_abstains_without_calling_the_llm():
    class Boom:
        name = "boom"

        def stream(self, *a, **k):
            raise AssertionError("LLM must not be called")

    *text, ans = stream_answer("zzz", Fixed([]), Boom())
    assert ans.abstained and ans.citations == [] and "".join(text) == ans.answer


def test_stream_answer_fails_loudly_if_the_backend_never_finishes():
    class Truncated:
        name = "cut"

        def stream(self, *a, **k):
            yield "Revenue"

    with pytest.raises(RuntimeError, match="without a final result"):
        list(stream_answer("q", Fixed([_hit(1)]), Truncated()))


# --- the real SDKs against a fake HTTP server (catches request shapes the installed SDK rejects) ---


def _sse_body(events: list[tuple[str | None, dict | str]]) -> bytes:
    out = []
    for name, data in events:
        payload = data if isinstance(data, str) else json.dumps(data)
        out.append((f"event: {name}\n" if name else "") + f"data: {payload}\n\n")
    return "".join(out).encode()


def test_anthropic_llm_streams_through_the_installed_sdk():
    from anthropic import Anthropic

    from filings_rag.llm import AnthropicLLM

    sent = []
    msg = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 21, "output_tokens": 1},
    }

    def ev(kind: str, **fields) -> tuple[str, dict]:
        return kind, {"type": kind, **fields}

    def text(t: str) -> tuple[str, dict]:
        return ev("content_block_delta", index=0, delta={"type": "text_delta", "text": t})

    body = _sse_body(
        [
            ev("message_start", message=msg),
            ev("content_block_start", index=0, content_block={"type": "text", "text": ""}),
            text("Revenue "),
            text("rose [1]."),
            ev("content_block_stop", index=0),
            ev("message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None}, usage={"output_tokens": 6}),
            ev("message_stop"),
        ]
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = Anthropic(api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    *text, final = AnthropicLLM("", "claude-haiku-4-5-20251001", client=client).stream("sys", "question", 100)
    assert text == ["Revenue ", "rose [1]."]
    assert final.text == "Revenue rose [1]." and (final.input_tokens, final.output_tokens) == (21, 6)
    assert final.first_token_ms is not None and final.first_token_ms <= final.latency_ms
    assert sent[0]["stream"] is True and sent[0]["system"] == "sys" and "temperature" not in sent[0]


def test_openai_compatible_llm_streams_strips_thinking_and_reads_usage():
    from openai import OpenAI, _base_client

    from filings_rag.llm import OpenAICompatibleLLM

    hx = getattr(_base_client, "httpx2", None) or _base_client.httpx

    def chunk(content=None, usage=None, choices=True):
        return {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen3:8b",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}] if choices else [],
            "usage": usage,
        }

    body = _sse_body(
        [
            (None, chunk("<think>")),
            (None, chunk("weigh the sources")),
            (None, chunk("</think>\n\n")),
            (None, chunk("Revenue rose")),
            (None, chunk(" [1].")),
            (None, chunk(choices=False, usage={"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59})),
            (None, "[DONE]"),
        ]
    )
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return hx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    llm = OpenAICompatibleLLM("http://x/v1", "qwen3:8b", "vllm", reasoning_tokens=100)
    llm.client = OpenAI(base_url="http://x/v1", api_key="k", http_client=hx.Client(transport=hx.MockTransport(handler)))
    *text, final = llm.stream("sys", "user", 700)
    assert "".join(text) == "Revenue rose [1]." == final.text
    assert (final.input_tokens, final.output_tokens) == (50, 9) and final.model == "vllm:qwen3:8b"
    assert final.first_token_ms is not None
    assert sent[0]["stream"] is True and sent[0]["stream_options"] == {"include_usage": True}
    assert sent[0]["max_tokens"] == 800


# --- /api/ask/stream ---


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _client(db, embedder, llm=None, **overrides) -> TestClient:
    s = Settings(rerank_enabled=False, rate_limit_per_minute=3, daily_question_cap=100, **overrides)
    app = create_app(s, conn=db, embedder=embedder, llm=llm or FakeLLM(), reranker=KeywordOverlapReranker())
    return TestClient(app)


def test_api_stream_sends_tokens_then_the_full_answer(db, embedder):
    with _client(db, embedder) as c:
        r = c.post("/api/ask/stream", json={"question": "What drove gross margin?", "tickers": ["ACME"]})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = _events(r.text)
        names = [n for n, _ in events]
        assert names[-1] == "answer" and set(names[:-1]) == {"token"} and len(names) > 2
        ans = events[-1][1]
        assert "".join(d["text"] for n, d in events[:-1]) == ans["answer"]
        assert ans["citations"][0]["section"].startswith("Item ") and "contexts" not in ans
        assert ans["timings_ms"]["first_token_ms"] >= 0

        plain = c.post("/api/ask", json={"question": "What drove gross margin?", "tickers": ["ACME"]}).json()
        assert plain["answer"] == ans["answer"] and plain["citations"] == ans["citations"]


def test_api_stream_validates_and_shares_the_rate_limit(db, embedder):
    with _client(db, embedder) as c:
        assert c.post("/api/ask/stream", json={"question": "x" * 600}).status_code == 422
        assert c.post("/api/ask/stream", json={"question": "ok?", "mode": "magic"}).status_code == 422
        codes = [c.post(p, json={"question": "revenue?"}).status_code for p in ["/api/ask", "/api/ask/stream"] * 2]
        assert codes == [200, 200, 200, 429]


def test_api_stream_reports_a_model_failure_as_an_error_event(db, embedder):
    class Dies:
        name = "dies"

        def generate(self, *a, **k):
            raise TimeoutError

        def stream(self, *a, **k):
            yield "Revenue"
            raise TimeoutError("model down")

    with _client(db, embedder, llm=Dies()) as c:
        events = _events(c.post("/api/ask/stream", json={"question": "revenue growth?"}).text)
    assert events[0] == ("token", {"text": "Revenue"})
    assert events[-1][0] == "error" and "TimeoutError" in events[-1][1]["detail"]


def test_api_stream_needs_a_streaming_backend(db, embedder):
    class OnlyGenerate:
        name = "plain"

        def generate(self, *a, **k):
            return Generation("x", "plain", 0, 0, 0.0)

    with _client(db, embedder, llm=OnlyGenerate()) as c:
        r = c.post("/api/ask/stream", json={"question": "revenue?"})
    assert r.status_code == 501 and "/api/ask" in r.json()["detail"]


def test_stream_answer_against_the_database(db, embedder):
    r = Retriever(db, embedder, KeywordOverlapReranker())
    *text, ans = stream_answer(
        "How much did revenue grow?", r, FakeLLM(), mode="hybrid", filters=Filters(tickers=["ACME"])
    )
    assert ans.citations[0].ticker == "ACME" and "".join(text) == ans.answer


def test_latency_benchmark_records_time_to_first_token(monkeypatch):
    import benchmarks.llm_latency as bl

    monkeypatch.setattr(bl, "get_settings", lambda: Settings(llm_provider="fake"))
    prompts = ["Sources:\n\n[1] Acme\nRevenue rose 12%. More.\n\nQuestion: q"]
    r = bl.bench("fake", prompts, runs=3, max_tokens=50)
    assert r["runs"] == 3 and r["model"] == "fake" and r["median_first_token_ms"] == 0.0
    row = bl.table_row("fake", r)
    assert row.startswith("| fake | fake | 0 ms |") and row.count("|") == 7
