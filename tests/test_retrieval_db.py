"""Retrieval against a real Postgres + pgvector database."""

from filings_rag.db import list_filings
from filings_rag.rerank import KeywordOverlapReranker
from filings_rag.retrieval import Filters, Retriever


def test_ingest_is_idempotent(db, embedder, tenk_text):
    from filings_rag.ingest import ingest_text
    from tests.conftest import make_filing

    before = db.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    added = ingest_text(db, embedder, make_filing("ACME", "Acme Robotics Inc.", "0000000001-25-000001"), tenk_text)
    assert added == 0
    assert db.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == before
    rows = list_filings(db)
    assert {r["ticker"] for r in rows} == {"ACME", "GLBX"}
    assert all(r["chunks"] > 0 for r in rows)


def test_only_selected_items_are_indexed(db):
    items = {r["section_item"] for r in db.execute("SELECT DISTINCT section_item FROM chunks").fetchall()}
    assert items == {"1", "1A", "1C", "7", "7A", "8"}


def test_each_mode_finds_the_cybersecurity_section(db, embedder):
    r = Retriever(db, embedder, KeywordOverlapReranker())
    q = "Who oversees cybersecurity risk and reports to the Audit Committee?"
    for mode in ("vector", "keyword", "hybrid", "hybrid_rerank"):
        res = r.search(q, mode=mode, k=3, filters=Filters(tickers=["ACME"]))
        assert res.hits, mode
        assert res.hits[0].section_item == "1C", (mode, [h.section_item for h in res.hits])
        assert all(h.ticker == "ACME" for h in res.hits)


def test_hybrid_fuses_both_rankings_with_rrf(db, embedder):
    r = Retriever(db, embedder)
    res = r.search("lidar sensor supplier", mode="hybrid", k=5)
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True)
    # A chunk found by both searches scores above the single-list maximum of 1/61
    assert scores[0] > 1 / 61


def test_filters_by_ticker_and_section(db, embedder):
    r = Retriever(db, embedder)
    res = r.search("interest rate risk", mode="hybrid", k=5, filters=Filters(tickers=["glbx"], section_items=["7a"]))
    assert res.hits and all(h.ticker == "GLBX" and h.section_item == "7A" for h in res.hits)
    assert res.hits[0].company == "Globex Bank Corp."
    assert "Globex Bank" in res.hits[0].source_label


def test_rerank_mode_falls_back_to_hybrid_without_a_reranker(db, embedder):
    res = Retriever(db, embedder, reranker=None).search("revenue growth", mode="hybrid_rerank", k=3)
    assert res.mode == "hybrid"
    assert "rerank_ms" not in res.timings_ms


def test_keyword_search_with_no_matches_returns_empty(db, embedder):
    res = Retriever(db, embedder).search("zzzqqqxx", mode="keyword", k=3)
    assert res.hits == []
