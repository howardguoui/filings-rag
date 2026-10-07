"""Run the real anthropic SDK and the real RAGAS judge against a fake HTTP server.

These catch the bug class unit fakes can't: passing an argument the installed SDK
doesn't accept (it raises TypeError before any request is sent).
"""

import asyncio
import json

import httpx2 as httpx
import pytest


def _message(text: str) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }


def test_anthropic_llm_calls_the_installed_sdk():
    from anthropic import Anthropic

    from filings_rag.llm import AnthropicLLM

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_message("Revenue grew [1]."))

    client = Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = AnthropicLLM("", "claude-haiku-4-5-20251001", client=client).generate("sys", "question", 100)
    assert out.text == "Revenue grew [1]."
    assert (out.input_tokens, out.output_tokens) == (12, 5)
    assert sent[0]["system"] == "sys" and sent[0]["max_tokens"] == 100


def test_ragas_judge_request_reaches_the_api():
    """The judge must get as far as an HTTP request; a TypeError means a bad model arg."""
    pytest.importorskip("ragas")
    from pydantic import BaseModel

    import evals.run_evals as ev
    from filings_rag.config import Settings

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}})

    import anthropic

    real = anthropic.AsyncAnthropic

    class Patched(real):  # a subclass so instructor's isinstance() checks still pass
        def __init__(self, **kw):
            super().__init__(**kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), max_retries=0)

    anthropic.AsyncAnthropic = Patched
    try:
        judge = ev.make_judge(Settings(anthropic_api_key="test"), "anthropic")
    finally:
        anthropic.AsyncAnthropic = real
    assert "temperature" not in judge.model_args and "top_p" not in judge.model_args

    class Verdict(BaseModel):
        ok: bool

    with pytest.raises(Exception) as err:
        asyncio.run(judge.agenerate("Is the sky blue?", Verdict))
    assert not isinstance(err.value, TypeError), err.value
    assert sent and "temperature" not in sent[0] and "top_p" not in sent[0]


def test_check_judged_fails_when_nothing_was_scored():
    import evals.run_evals as ev

    rows = [{"faithfulness": None, "answer_relevancy": None, "context_precision": None, "errors": ["boom"]}]
    with pytest.raises(RuntimeError, match="boom"):
        ev.check_judged(rows)
    ev.check_judged([{"faithfulness": 0.9, "answer_relevancy": None, "context_precision": None}])


def test_openai_compatible_judge_and_llm_send_room_to_think():
    """A thinking model on Ollama/vLLM needs a larger output limit, or answers and judge verdicts are cut off."""
    pytest.importorskip("ragas")
    import openai
    from openai import OpenAI, _base_client
    from pydantic import BaseModel

    import evals.run_evals as ev
    from filings_rag.config import Settings
    from filings_rag.llm import OpenAICompatibleLLM

    # the HTTP library this openai version was built on: httpx (openai 1.x) or httpx2 (newer)
    hx = getattr(_base_client, "httpx2", None) or _base_client.httpx
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return hx.Response(400, json={"error": {"message": "x"}})

    real = openai.AsyncOpenAI

    class Patched(real):
        def __init__(self, **kw):
            super().__init__(**kw, http_client=hx.AsyncClient(transport=hx.MockTransport(handler)), max_retries=0)

    openai.AsyncOpenAI = Patched
    try:
        judge = ev.make_judge(Settings(judge_max_tokens=8192), "ollama")
    finally:
        openai.AsyncOpenAI = real

    class Verdict(BaseModel):
        ok: bool

    with pytest.raises(Exception) as err:
        asyncio.run(judge.agenerate("Is the sky blue?", Verdict))
    assert not isinstance(err.value, TypeError), err.value
    assert sent and sent[0].get("max_tokens") == 8192

    llm = OpenAICompatibleLLM("http://x/v1", "qwen3:8b", "ollama", reasoning_tokens=3072)
    llm.client = OpenAI(
        base_url="http://x/v1",
        api_key="k",
        max_retries=0,
        http_client=hx.Client(transport=hx.MockTransport(handler)),
    )
    with pytest.raises(openai.BadRequestError):
        llm.generate("sys", "user", 700)
    assert sent[-1]["max_tokens"] == 700 + 3072
