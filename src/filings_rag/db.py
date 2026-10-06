"""Postgres + pgvector storage: one table of filings, one of chunks.

Chunks hold both an embedding (HNSW index, cosine distance) and a generated
full-text column (GIN index), so one query can run vector and keyword search.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .chunking import Chunk
from .edgar import Filing


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url, row_factory=dict_row, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    _configure(conn)
    return conn


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    # pgvector >= 0.8: when a WHERE filter (ticker, year) discards most HNSW candidates,
    # keep walking the graph until LIMIT rows match instead of returning fewer than k.
    try:
        conn.execute("SET hnsw.iterative_scan = strict_order")
    except psycopg.errors.Error:
        pass  # older pgvector; filtered searches may return fewer rows


def make_pool(database_url: str, dim: int, max_size: int = 4):
    """Connection pool for the web app: checks each connection before use and replaces
    dead ones, so the demo survives a database restart or an idle-connection drop."""
    from psycopg_pool import ConnectionPool

    with connect(database_url) as setup:  # extension + tables must exist before pooled connections register types
        init_schema(setup, dim)
    return ConnectionPool(
        database_url,
        min_size=1,
        max_size=max_size,
        kwargs={"row_factory": dict_row, "autocommit": True},
        configure=_configure,
        check=ConnectionPool.check_connection,
        timeout=15,
        open=True,
    )


def init_schema(conn: psycopg.Connection, dim: int) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS filings (
            id          SERIAL PRIMARY KEY,
            ticker      TEXT NOT NULL,
            company     TEXT NOT NULL,
            cik         INTEGER NOT NULL,
            form        TEXT NOT NULL,
            accession   TEXT NOT NULL UNIQUE,
            filing_date DATE NOT NULL,
            report_date DATE NOT NULL,
            fiscal_year INTEGER NOT NULL,
            url         TEXT NOT NULL
        )"""
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id            BIGSERIAL PRIMARY KEY,
            filing_id     INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
            section_item  TEXT NOT NULL,
            section_title TEXT NOT NULL,
            ordinal       INTEGER NOT NULL,
            header        TEXT NOT NULL,
            text          TEXT NOT NULL,
            embedding     vector({dim}) NOT NULL,
            tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', header || ' ' || text)) STORED
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_filing ON chunks (filing_id)")


def has_chunks(conn: psycopg.Connection, accession: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM chunks c JOIN filings f ON f.id = c.filing_id WHERE f.accession = %s LIMIT 1", (accession,)
    ).fetchone()
    return row is not None


def upsert_filing(conn: psycopg.Connection, f: Filing) -> tuple[int, bool]:
    """Insert a filing row. Returns (id, already_had_chunks)."""
    row = conn.execute(
        """
        INSERT INTO filings (ticker, company, cik, form, accession, filing_date, report_date, fiscal_year, url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (accession) DO UPDATE SET company = EXCLUDED.company
        RETURNING id""",
        (f.ticker, f.company, f.cik, f.form, f.accession, f.filing_date, f.report_date, f.fiscal_year, f.url),
    ).fetchone()
    has = conn.execute("SELECT 1 FROM chunks WHERE filing_id = %s LIMIT 1", (row["id"],)).fetchone()
    return row["id"], has is not None


def insert_chunks(
    conn: psycopg.Connection, filing_id: int, chunks: Iterable[Chunk], vectors: Iterable[np.ndarray]
) -> int:
    rows = [
        (filing_id, c.section_item, c.section_title, c.ordinal, c.header, c.text, v)
        for c, v in zip(chunks, vectors, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (filing_id, section_item, section_title, ordinal, header, text, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            rows,
        )
    return len(rows)


def list_filings(conn: psycopg.Connection) -> list[dict]:
    return conn.execute(
        """SELECT f.ticker, f.company, f.fiscal_year, f.filing_date, f.url, count(c.id) AS chunks
           FROM filings f LEFT JOIN chunks c ON c.filing_id = f.id
           GROUP BY f.id ORDER BY f.ticker, f.fiscal_year DESC"""
    ).fetchall()
