"""Answer-generation backends behind one interface.

- anthropic: Claude API (used by the public demo; set ANTHROPIC_API_KEY and a spend limit)
- ollama:    local models through Ollama's OpenAI-compatible endpoint
- vllm:      a vLLM server (OpenAI-compatible), e.g. on an RTX GPU in WSL
- fake:      deterministic stub for tests
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from .config import Settings


@dataclass
class Generation:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    first_token_ms: float | None = None  # set by stream(): when the first visible text arrived


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str, max_tokens: int) -> Generation: ...


class StreamingLLM(LLM, Protocol):
    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | Generation]:
        """Yield answer text as it arrives, then one final Generation with the full text and timings."""
        ...


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


class ThinkFilter:
    """Drop a leading <think>...</think> block from streamed text, matching what generate() strips.

    Ollama sends reasoning in a separate field, but other OpenAI-compatible servers (vLLM without a
    reasoning parser) put it inline in the content, so the first chunks are held back until it's clear.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.passing = False

    def feed(self, piece: str) -> str:
        if self.passing:
            return piece
        self.buf += piece
        head = self.buf.lstrip()
        if self.CLOSE in head:
            out = head.split(self.CLOSE, 1)[1].lstrip()
        elif head.startswith(self.OPEN) or self.OPEN.startswith(head):
            return ""  # inside (or possibly starting) a think block: keep waiting
        else:
            out = head
        self.passing, self.buf = True, ""
        return out

    def flush(self) -> str:
        """End of stream. An unclosed think block means the model ran out of tokens while reasoning: no answer."""
        head, self.buf = self.buf.lstrip(), ""
        return "" if head.startswith(self.OPEN) or not head else head


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, client=None):
        from anthropic import Anthropic

        if client is None and not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = client or Anthropic(api_key=api_key)
        self.name = model

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        t0 = time.perf_counter()
        msg = self.client.messages.create(
            model=self.name,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return Generation(
            text=text,
            model=self.name,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | Generation]:
        t0 = time.perf_counter()
        first = None
        parts: list[str] = []
        with self.client.messages.stream(
            model=self.name,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                if not text:
                    continue
                if first is None:
                    first = _ms(t0)
                parts.append(text)
                yield text
            msg = stream.get_final_message()
        yield Generation(
            text="".join(parts),
            model=self.name,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_ms=_ms(t0),
            first_token_ms=first,
        )


class OpenAICompatibleLLM:
    """Works with Ollama (http://localhost:11434/v1) and vLLM (http://localhost:8000/v1)."""

    def __init__(self, base_url: str, model: str, label: str, reasoning_tokens: int = 0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.name = model
        self.label = label
        self.reasoning_tokens = reasoning_tokens  # added to max_tokens: room for a reasoning model to think

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.name,
            max_tokens=max_tokens + self.reasoning_tokens,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = resp.usage
        text = resp.choices[0].message.content or ""
        # Qwen3 and other reasoning models may wrap their thinking in <think> tags
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return Generation(
            text=text.strip(),
            model=f"{self.label}:{self.name}",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | Generation]:
        t0 = time.perf_counter()
        first = None
        usage = None
        parts: list[str] = []
        think = ThinkFilter()
        chunks = self.client.chat.completions.create(
            model=self.name,
            max_tokens=max_tokens + self.reasoning_tokens,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            stream=True,
            stream_options={"include_usage": True},  # final chunk carries token counts (Ollama and vLLM)
        )
        for chunk in chunks:
            usage = chunk.usage or usage
            delta = chunk.choices[0].delta.content if chunk.choices else None
            text = think.feed(delta) if delta else ""
            if text:
                first = first if first is not None else _ms(t0)
                parts.append(text)
                yield text
        tail = think.flush()
        if tail:
            first = first if first is not None else _ms(t0)
            parts.append(tail)
            yield tail
        yield Generation(
            text="".join(parts).strip(),
            model=f"{self.label}:{self.name}",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=_ms(t0),
            first_token_ms=first,
        )


class FakeLLM:
    """Answers with the first sentence of the first source and cites it. For tests."""

    name = "fake"

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        marker = "[1] "
        start = user.find(marker)
        if start == -1:
            return Generation("The filings provided do not contain this information.", "fake", 0, 0, 0.0)
        # Only source [1]: a table source has no ". " to stop at, and reading on into "[2] ..." would cite [2].
        body = user[start + len(marker) :].split("\n\n[2] ")[0].split("\n", 2)[-1]
        first = body.split(". ")[0].strip()
        return Generation(f"{first} [1].", "fake", 0, 0, 0.0)

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | Generation]:
        gen = self.generate(system, user, max_tokens)
        words = gen.text.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "
        gen.first_token_ms = 0.0
        yield gen


def make_llm(settings: Settings, provider: str | None = None) -> LLM:
    p = provider or settings.llm_provider
    if p == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    if p == "ollama":
        return OpenAICompatibleLLM(
            settings.ollama_base_url.rstrip("/") + "/v1", settings.ollama_model, "ollama", settings.reasoning_tokens
        )
    if p == "vllm":
        return OpenAICompatibleLLM(settings.vllm_base_url, settings.vllm_model, "vllm", settings.reasoning_tokens)
    return FakeLLM()
