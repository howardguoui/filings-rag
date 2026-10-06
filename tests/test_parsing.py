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
