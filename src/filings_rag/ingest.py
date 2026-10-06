"""Ingestion pipeline: EDGAR -> text -> Items -> chunks -> embeddings -> Postgres."""

from __future__ import annotations

import logging
import time

import psycopg

from .chunking import chunk_section
from .db import init_schema, insert_chunks, upsert_filing
from .edgar import EdgarClient, Filing
from .embeddings import Embedder
from .sections import split_items

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
    """Chunk, embed and store one filing's text. Skips filings already stored."""
    filing_id, already = upsert_filing(conn, filing)
    if already:
        log.info("skip %s FY%s (already ingested)", filing.ticker, filing.fiscal_year)
        return 0
    doc_header = f"{filing.company} ({filing.ticker}) {filing.form} FY{filing.fiscal_year}"
    chunks = []
    for section in split_items(text):
        if items and section.item not in items:
            continue
        chunks.extend(chunk_section(section, doc_header))
    if not chunks:
        log.warning("no chunks for %s %s", filing.ticker, filing.accession)
        return 0
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
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
