"""Answer-generation backends behind one interface.

- anthropic: Claude API (used by the public demo; set ANTHROPIC_API_KEY and a spend limit)
- ollama:    local models through Ollama's OpenAI-compatible endpoint
- vllm:      a vLLM server (OpenAI-compatible), e.g. on an RTX GPU in WSL
- fake:      deterministic stub for tests
"""

from __future__ import annotations

import time
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


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str, max_tokens: int) -> Generation: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = Anthropic(api_key=api_key)
        self.name = model

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        t0 = time.perf_counter()
        msg = self.client.messages.create(
            model=self.name,
            max_tokens=max_tokens,
            temperature=0,
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


class OpenAICompatibleLLM:
    """Works with Ollama (http://localhost:11434/v1) and vLLM (http://localhost:8000/v1)."""

    def __init__(self, base_url: str, model: str, label: str):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.name = model
        self.label = label

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.name,
            max_tokens=max_tokens,
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


class FakeLLM:
    """Answers with the first sentence of the first source and cites it. For tests."""

    name = "fake"

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        marker = "[1] "
        start = user.find(marker)
        if start == -1:
            return Generation("The filings provided do not contain this information.", "fake", 0, 0, 0.0)
        body = user[start + len(marker) :].split("\n", 2)[-1]
        first = body.split(". ")[0].strip()
        return Generation(f"{first} [1].", "fake", 0, 0, 0.0)


def make_llm(settings: Settings, provider: str | None = None) -> LLM:
    p = provider or settings.llm_provider
    if p == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    if p == "ollama":
        return OpenAICompatibleLLM(settings.ollama_base_url.rstrip("/") + "/v1", settings.ollama_model, "ollama")
    if p == "vllm":
        return OpenAICompatibleLLM(settings.vllm_base_url, settings.vllm_model, "vllm")
    return FakeLLM()
