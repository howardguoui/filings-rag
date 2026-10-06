"""Retrieval: vector, keyword, hybrid (Reciprocal Rank Fusion), and hybrid + cross-encoder rerank.

Hybrid search runs both searches inside Postgres and fuses the two rankings with
RRF: score = sum(1 / (60 + rank)). Keyword search catches exact terms such as
"Item 1C" or "Basel III" that embeddings blur; vector search catches paraphrases.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import psycopg

from .config import RetrievalMode
from .embeddings import Embedder

RRF_K = 60


@dataclass
class Hit:
    id: int
    ticker: str
    company: str
    fiscal_year: int
    section_item: str
    section_title: str
    header: str
    text: str
    url: str
    score: float

    @property
    def source_label(self) -> str:
        return (
            f"{self.company} ({self.ticker}) 10-K FY{self.fiscal_year}, Item {self.section_item}. {self.section_title}"
        )


@dataclass
class Filters:
    tickers: list[str] | None = None
    fiscal_year: int | None = None
    section_items: list[str] | None = None

    def sql(self) -> tuple[str, dict]:
        clauses, params = [], {}
        if self.tickers:
            clauses.append("f.ticker = ANY(%(tickers)s)")
            params["tickers"] = [t.upper() for t in self.tickers]
        if self.fiscal_year:
            clauses.append("f.fiscal_year = %(fy)s")
            params["fy"] = self.fiscal_year
        if self.section_items:
            clauses.append("c.section_item = ANY(%(items)s)")
            params["items"] = [s.upper() for s in self.section_items]
        return (" AND " + " AND ".join(clauses)) if clauses else "", params


@dataclass
class RetrievalResult:
    hits: list[Hit]
    mode: str
    timings_ms: dict[str, float] = field(default_factory=dict)


_SELECT = """c.id, f.ticker, f.company, f.fiscal_year, c.section_item, c.section_title,
             c.header, c.text, f.url"""


def _vector_sql(where: str) -> str:
    return f"""
        SELECT {_SELECT}, 1 - (c.embedding <=> %(q)s) AS score
        FROM chunks c JOIN filings f ON f.id = c.filing_id
        WHERE TRUE {where}
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s"""


def _keyword_sql(where: str) -> str:
    return f"""
        SELECT {_SELECT}, ts_rank_cd(c.tsv, query) AS score
        FROM chunks c JOIN filings f ON f.id = c.filing_id,
             websearch_to_tsquery('english', %(text)s) query
        WHERE c.tsv @@ query {where}
        ORDER BY score DESC
        LIMIT %(k)s"""


def _hybrid_sql(where: str) -> str:
    return f"""
        WITH v AS (
            SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> %(q)s) AS r
            FROM chunks c JOIN filings f ON f.id = c.filing_id
            WHERE TRUE {where}
            ORDER BY c.embedding <=> %(q)s
            LIMIT %(cand)s
        ),
        kw AS (
            SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, query) DESC) AS r
            FROM chunks c JOIN filings f ON f.id = c.filing_id,
                 websearch_to_tsquery('english', %(text)s) query
            WHERE c.tsv @@ query {where}
            ORDER BY ts_rank_cd(c.tsv, query) DESC
            LIMIT %(cand)s
        ),
        fused AS (
            SELECT COALESCE(v.id, kw.id) AS id,
                   COALESCE(1.0 / ({RRF_K} + v.r), 0) + COALESCE(1.0 / ({RRF_K} + kw.r), 0) AS score
            FROM v FULL OUTER JOIN kw ON v.id = kw.id
        )
        SELECT {_SELECT}, fused.score
        FROM fused JOIN chunks c ON c.id = fused.id JOIN filings f ON f.id = c.filing_id
        ORDER BY fused.score DESC
        LIMIT %(k)s"""


class Retriever:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder, reranker=None, candidate_k: int = 40):
        self.conn = conn
        self.embedder = embedder
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(
        self, query: str, mode: RetrievalMode = "hybrid", k: int = 6, filters: Filters | None = None
    ) -> RetrievalResult:
        filters = filters or Filters()
        where, params = filters.sql()
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        if mode == "hybrid_rerank" and self.reranker is None:
            mode = "hybrid"  # reranker disabled in this deployment
        fetch_k = self.candidate_k if mode == "hybrid_rerank" else k

        params.update({"text": query, "k": fetch_k, "cand": self.candidate_k})
        if mode in ("vector", "hybrid", "hybrid_rerank"):
            params["q"] = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
            timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        sql = {"vector": _vector_sql, "keyword": _keyword_sql}.get(mode, _hybrid_sql)(where)
        t1 = time.perf_counter()
        rows = self.conn.execute(sql, params).fetchall()
        timings["search_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        hits = [Hit(**{**r, "score": float(r["score"])}) for r in rows]

        if mode == "hybrid_rerank" and hits:
            t2 = time.perf_counter()
            hits = self.reranker.rerank(query, hits, k)
            timings["rerank_ms"] = round((time.perf_counter() - t2) * 1000, 1)
        return RetrievalResult(hits=hits[:k], mode=mode, timings_ms=timings)
