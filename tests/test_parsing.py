import pytest

from filings_rag.chunking import chunk_section
from filings_rag.edgar import EdgarClient, filings_from_submissions, html_to_text
from filings_rag.sections import Section, split_items


def test_html_to_text_drops_hidden_xbrl_and_keeps_table_rows(tenk_text):
    assert "hidden xbrl header" not in tenk_text
    assert "Robots | Fiscal" not in tenk_text  # header row is separate
    assert "Services | 700 | 534 | 31" in tenk_text  # lone $ and % cells dropped, numbers kept
    assert "\n\n\n" not in tenk_text


def test_html_to_text_handles_nbsp_and_blocks():
    out = html_to_text("<html><body><div>Item&nbsp;7.&nbsp;MD&amp;A</div><p>Line one</p><p>Line two</p></body></html>")
    assert out.splitlines() == ["Item 7. MD&A", "Line one", "Line two"]


def test_split_items_takes_real_sections_not_table_of_contents(tenk_text):
    sections = {s.item: s for s in split_items(tenk_text)}
    assert {"1", "1A", "1C", "7", "7A", "8"} <= set(sections)
    assert "single supplier for the lidar sensors" in sections["1A"].text
    assert "Audit Committee" in sections["1C"].text
    assert sections["7A"].title.startswith("Quantitative")
    # The Item 1A section must not swallow the cybersecurity section
    assert "Chief Information Security Officer" not in sections["1A"].text


def test_split_items_without_headings_returns_whole_document():
    sections = split_items("Just some text without any item headings. " * 20)
    assert len(sections) == 1 and sections[0].item == "0"


def test_chunk_section_sizes_header_and_overlap():
    para = " ".join(f"word{i}" for i in range(60)) + "."
    sec = Section(item="7", title="Management's Discussion and Analysis", text="\n".join([para] * 12))
    chunks = chunk_section(sec, "Acme (ACME) 10-K FY2025", target_words=200, overlap_words=20)
    assert len(chunks) >= 3
    assert all(len(c.text.split()) <= 200 + 20 + 60 for c in chunks)
    assert chunks[0].header == "Acme (ACME) 10-K FY2025, Item 7. Management's Discussion and Analysis"
    assert chunks[0].embed_text.startswith(chunks[0].header)
    # overlap: the next chunk starts with the tail of the previous one
    assert chunks[0].text.split()[-1] == chunks[1].text.split()[19]
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunk_section_splits_one_giant_paragraph():
    sec = Section(item="1A", title="Risk Factors", text="A risk sentence here. " * 400)
    chunks = chunk_section(sec, "X", target_words=150, overlap_words=0)
    assert len(chunks) > 5
    assert max(len(c.text.split()) for c in chunks) <= 150


def test_filings_from_submissions_skips_amendments_and_other_forms():
    subs = {
        "name": "Acme Robotics Inc.",
        "filings": {
            "recent": {
                "form": ["10-Q", "10-K/A", "10-K", "8-K", "10-K"],
                "accessionNumber": ["a", "b", "0001-25-000010", "d", "0001-24-000009"],
                "filingDate": ["2025-12-01", "2025-11-20", "2025-11-01", "2025-10-01", "2024-11-01"],
                "reportDate": ["2025-11-30", "2025-09-30", "2025-09-30", "", "2024-09-30"],
                "primaryDocument": ["q.htm", "a.htm", "k25.htm", "8k.htm", "k24.htm"],
            }
        },
    }
    out = filings_from_submissions(subs, "ACME", "Acme", 320193, 2)
    assert [f.fiscal_year for f in out] == [2025, 2024]
    assert out[0].url == "https://www.sec.gov/Archives/edgar/data/320193/000125000010/k25.htm"


def test_edgar_client_requires_contact_email():
    with pytest.raises(ValueError, match="contact email"):
        EdgarClient("my-script")


def test_item_headings_ignore_prose_cross_references():
    from filings_rag.sections import split_items

    body = "Real risk discussion. " * 20
    text = (
        "Item 1A. Risk Factors\n" + body + "\n"
        "Item 7 of Part II, Management's Discussion and Analysis, explains how these risks affected results "
        "during the year and should be read together with the financial statements in Item 8.\n" + body + "\n"
        "Item 7. Management's Discussion and Analysis\n" + body
    )
    sections = {s.item: s.text for s in split_items(text)}
    assert set(sections) == {"1A", "7"}
    assert "Item 7 of Part II" in sections["1A"]  # the cross-reference stayed inside Risk Factors


def test_cross_reference_index_10k_is_indexed_whole(db, embedder):
    from filings_rag.ingest import ingest_text
    from filings_rag.sections import looks_itemized, split_items
    from tests.conftest import make_filing

    text = (
        "Form 10-K Cross-reference Index\n"
        "Item 1A. | Risk Factors | 8-31\n"
        "Item 7. | Management's Discussion and Analysis | 52-160\n"
        "Item 16. | Form 10-K Summary | None\n"
        + ("Risk Factors\nCredit risk is the risk of loss from obligor default. " * 40)
    )
    assert not looks_itemized(split_items(text))
    n = ingest_text(db, embedder, make_filing("XREF", "Crossref Bank", "0000000003-25-000001"), text)
    assert n > 0
    rows = db.execute(
        "SELECT DISTINCT c.section_item, c.header FROM chunks c JOIN filings f ON f.id = c.filing_id "
        "WHERE f.ticker = 'XREF'"
    ).fetchall()
    assert [r["section_item"] for r in rows] == ["0"]
    assert rows[0]["header"].endswith("FY2025, Full document")


def _recent(forms, years):
    return {
        "form": forms,
        "accessionNumber": [f"0001-{y % 100}-{i:06d}" for i, y in enumerate(years)],
        "filingDate": [f"{y}-02-15" for y in years],
        "reportDate": [f"{y - 1}-12-31" for y in years],
        "primaryDocument": [f"d{i}.htm" for i in range(len(forms))],
    }


def test_latest_10ks_follows_paginated_submission_files(tmp_path, monkeypatch):
    import httpx

    import filings_rag.edgar as edgar

    pages = {
        "/files/company_tickers.json": {"0": {"cik_str": 19617, "ticker": "BANK", "title": "Big Bank"}},
        "/submissions/CIK0000019617.json": {
            "name": "Big Bank",
            "filings": {
                "recent": _recent(["8-K", "10-K", "424B2"], [2026, 2026, 2025]),
                "files": [{"name": "CIK0000019617-submissions-001.json"}],
            },
        },
        "/submissions/CIK0000019617-submissions-001.json": _recent(["10-K", "8-K", "10-K"], [2025, 2024, 2024]),
    }
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(503, headers={"retry-after": "0"})  # one transient failure
        return httpx.Response(200, json=pages[request.url.path])

    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    client = EdgarClient("Test Person test@example.com", cache_dir=tmp_path)
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    out = client.latest_10ks("BANK", count=3)
    assert [f.fiscal_year for f in out] == [2025, 2024, 2023]
    assert calls[0] == calls[1] == "/files/company_tickers.json"  # retried after the 503
    client.latest_10ks("BANK", count=3)
    assert len(calls) == 4  # second call served from the disk cache


def test_nested_tables_are_not_duplicated():
    from filings_rag.edgar import html_to_text

    html = (
        "<body><table><tr><td>Outer A</td><td><table><tr><td>Inner 1</td><td>Inner 2</td></tr></table></td></tr>"
        "<tr><td>Outer B</td><td>5</td></tr></table></body>"
    )
    assert html_to_text(html) == "Outer A | Inner 1 Inner 2\nOuter B | 5"


def test_filing_cache_is_utf8_whatever_the_platform_default(tmp_path):
    # 10-K cover pages use check-box characters ("☒") that Windows' default cp1252 can't encode;
    # the first real ingest on Windows crashed writing the cache. -X warn_default_encoding turns
    # any read or write that relies on the platform default into an error.
    import subprocess
    import sys

    code = (
        "import sys, httpx; from filings_rag.edgar import EdgarClient, Filing;"
        "c = EdgarClient('Test Person test@example.com', cache_dir=sys.argv[1]);"
        "c.http = httpx.Client(transport=httpx.MockTransport("
        "lambda r: httpx.Response(200, text='<p>☒ Annual report</p>')));"
        "f = Filing('X', 'X Corp', 1, '10-K', '0001-25-000001', '2025-01-01', '2024-12-31', 2024, 'x.htm');"
        "assert '☒' in c.filing_text(f); assert c.filing_text(f) == c.filing_text(f)"
    )
    subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning", "-c", code, str(tmp_path)],
        check=True,
    )
