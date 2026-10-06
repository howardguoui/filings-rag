"""Embedding providers. Ingestion and queries must use the same one.

- fastembed (default): BAAI/bge-small-en-v1.5, 384 dims, runs on CPU via ONNX, no API key.
  Small enough for a 512 MB free-tier web service.
- ollama: e.g. nomic-embed-text (768 dims) on your own GPU.
- hash: deterministic bag-of-words hashing, for tests and offline runs only.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx
import numpy as np

from .config import Settings

# bge models are trained with this instruction prefix on the query side
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    def __init__(self, model: str, dim: int):
        from fastembed import TextEmbedding

        self.model_name = model
        self.dim = dim
        self._model = TextEmbedding(model_name=model)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> np.ndarray:
        prefix = _BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return np.asarray(next(iter(self._model.embed([prefix + text]))), dtype=np.float32)


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, dim: int):
        self.url = base_url.rstrip("/") + "/api/embed"
        self.model = model
        self.dim = dim
        self.http = httpx.Client(timeout=120)

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        r = self.http.post(self.url, json={"model": self.model, "input": texts})
        r.raise_for_status()
        return [np.asarray(v, dtype=np.float32) for v in r.json()["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), 32):
            out.extend(self._embed(texts[i : i + 32]))
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class HashEmbedder:
    """Feature-hashed bag of words. Not semantic; exists so tests run without model downloads."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = math.sqrt(float((v * v).sum())) or 1.0
        return v / n

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def make_embedder(settings: Settings) -> Embedder:
    if settings.embed_provider == "fastembed":
        return FastEmbedEmbedder(settings.embed_model, settings.embed_dim)
    if settings.embed_provider == "ollama":
        return OllamaEmbedder(settings.ollama_base_url, settings.ollama_embed_model, settings.embed_dim)
    return HashEmbedder(settings.embed_dim)
