"""Ingestion pipeline: EDGAR -> text -> Items -> chunks -> embeddings -> Postgres."""

from __future__ import annotations

import logging
import time

import psycopg

from .chunking import chunk_section
from .db import has_chunks, init_schema, insert_chunks, upsert_filing
from .edgar import EdgarClient, Filing
from .embeddings import Embedder
from .sections import FULL_DOCUMENT, Section, looks_itemized, split_items

log = logging.getLogger(__name__)

# Items worth indexing for Q&A; the rest are boilerplate or "incorporated by reference".
DEFAULT_ITEMS = {"1", "1A", "1C", "2", "3", "5", "7", "7A", "8", "9A"}


def ingest_text(
    conn: psycopg.Connection,
    embedder: Embedder,
    filing: Filing,
    text: str,
    items: set[str] | None = DEFAULT_ITEMS,
) -> int:
    """Chunk, embed and store one filing's text. Skips filings already stored.

    The filing row and its chunks are written in one transaction, so an interrupted
    run never leaves a filing with half its chunks (which would then be skipped).
    """
    if has_chunks(conn, filing.accession):
        log.info("skip %s FY%s (already ingested)", filing.ticker, filing.fiscal_year)
        return 0
    doc_header = f"{filing.company} ({filing.ticker}) {filing.form} FY{filing.fiscal_year}"
    sections = split_items(text)
    if looks_itemized(sections):
        sections = [s for s in sections if not items or s.item in items]
    else:
        log.warning(
            "%s FY%s has no usable Item headings (cross-reference index?); indexing the whole document",
            filing.ticker,
            filing.fiscal_year,
        )
        sections = [Section(item=FULL_DOCUMENT, title="Full document", text=text.strip())]
    chunks = [c for section in sections for c in chunk_section(section, doc_header)]
    if not chunks:
        log.warning("no chunks for %s %s", filing.ticker, filing.accession)
        return 0
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    with conn.transaction():
        filing_id, _ = upsert_filing(conn, filing)
        return insert_chunks(conn, filing_id, chunks, vectors)


def ingest_tickers(
    conn: psycopg.Connection,
    embedder: Embedder,
    edgar: EdgarClient,
    tickers: list[str],
    years: int = 1,
) -> dict[str, int]:
    init_schema(conn, embedder.dim)
    counts: dict[str, int] = {}
    for ticker in tickers:
        for filing in edgar.latest_10ks(ticker, count=years):
            t0 = time.perf_counter()
            text = edgar.filing_text(filing)
            n = ingest_text(conn, embedder, filing, text)
            counts[f"{filing.ticker} FY{filing.fiscal_year}"] = n
            log.info("%s FY%s: %d chunks in %.1fs", filing.ticker, filing.fiscal_year, n, time.perf_counter() - t0)
    return counts
