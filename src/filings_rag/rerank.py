"""Cross-encoder reranking: scores each (question, passage) pair jointly, which is
slower than embeddings but much better at ordering the top candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import Hit


class CrossEncoderReranker:
    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        scores = list(self._model.rerank(query, [f"{h.header}\n{h.text}" for h in hits]))
        for h, s in zip(hits, scores, strict=True):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]


class KeywordOverlapReranker:
    """Lightweight stand-in for tests: ranks by shared query terms."""

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        terms = {t for t in query.lower().split() if len(t) > 3}
        for h in hits:
            body = h.text.lower()
            h.score = float(sum(body.count(t) for t in terms))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]
