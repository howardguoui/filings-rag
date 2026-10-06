"""Shared fixtures. Database tests need Postgres with pgvector:

    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/filings_test pytest

They are skipped when TEST_DATABASE_URL is not set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from filings_rag.edgar import Filing, html_to_text
from filings_rag.embeddings import HashEmbedder

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # no usage telemetry from test runs

FIXTURES = Path(__file__).parent / "fixtures"


def make_filing(ticker: str, company: str, accession: str, year: int = 2025) -> Filing:
    return Filing(
        ticker=ticker,
        company=company,
        cik=1000 + len(ticker),
        form="10-K",
        accession=accession,
        filing_date=f"{year}-11-01",
        report_date=f"{year}-09-30",
        fiscal_year=year,
        primary_doc="doc.htm",
    )


@pytest.fixture(scope="session")
def tenk_text() -> str:
    return html_to_text((FIXTURES / "sample_10k.html").read_text())


@pytest.fixture(scope="session")
def embedder() -> HashEmbedder:
    return HashEmbedder(384)


@pytest.fixture()
def db(embedder, tenk_text):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set (needs Postgres with pgvector)")
    from filings_rag.db import connect, init_schema
    from filings_rag.ingest import ingest_text

    conn = connect(url)
    conn.execute("DROP TABLE IF EXISTS chunks, filings CASCADE")
    init_schema(conn, embedder.dim)
    # Two fictional companies built from the same fixture, so filters can be tested
    ingest_text(conn, embedder, make_filing("ACME", "Acme Robotics Inc.", "0000000001-25-000001"), tenk_text)
    other = tenk_text.replace("Acme Robotics", "Globex Bank").replace("robot", "loan")
    ingest_text(conn, embedder, make_filing("GLBX", "Globex Bank Corp.", "0000000002-25-000002"), other)
    yield conn
    conn.execute("DROP TABLE IF EXISTS chunks, filings CASCADE")
    conn.close()
