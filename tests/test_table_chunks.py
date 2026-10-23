"""Financial tables become chunks of their own, led by their caption (see chunking.py)."""

from filings_rag.chunking import chunk_section
from filings_rag.sections import Section, split_items

HEADER = "Acme (ACME) 10-K FY2025"
PROSE = "Revenue grew because more robots shipped to grocery customers in every region we serve. " * 4


def _section(*lines: str) -> Section:
    return Section(item="7", title="Management's Discussion and Analysis", text="\n".join(lines))


def test_table_is_one_chunk_with_its_caption_between_the_prose_around_it():
    sec = _section(
        PROSE,
        "The following table shows net sales by segment (in millions):",
        "Segment | Fiscal 2025 | Fiscal 2024 | Change",
        "Robots | 1,700 | 1,480 | 15",
        "Services | 700 | 534 | 31",
        "Gross margin fell to 41% from 44% because of higher lidar sensor costs and expedited freight.",
    )
    chunks = chunk_section(sec, HEADER)
    assert [c.is_table for c in chunks] == [False, True, False]
    table = chunks[1]
    assert table.text.splitlines() == [
        "The following table shows net sales by segment (in millions):",
        "Segment | Fiscal 2025 | Fiscal 2024 | Change",
        "Robots | 1,700 | 1,480 | 15",
        "Services | 700 | 534 | 31",
    ]
    assert table.embed_text.startswith(HEADER)
    assert "Gross margin fell" in chunks[2].text and "Robots | 1,700" not in chunks[2].text
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_table_used_to_be_cut_across_prose_chunks():
    # The failure this fixes: with prose filling the budget, the table's rows landed in two chunks
    # and the second one had no header row or caption saying what the numbers were.
    rows = [f"Product line {i} | {1000 + i} | {900 + i} | {i}" for i in range(30)]
    sec = _section(PROSE * 3, "Net sales by product line (in millions):", "Line | 2025 | 2024 | Change", *rows)
    tables = [c for c in chunk_section(sec, HEADER, target_words=200) if c.is_table]
    assert all(
        c.text.startswith("Net sales by product line (in millions):\nLine | 2025 | 2024 | Change") for c in tables
    )


def test_large_table_is_split_by_rows_and_every_piece_repeats_caption_and_header_row():
    rows = [f"Product line {i} | {1000 + i} | {900 + i} | {i}" for i in range(80)]
    sec = _section("Net sales by product line (in millions):", "Line | 2025 | 2024 | Change", *rows)
    chunks = chunk_section(sec, HEADER, target_words=120)
    assert len(chunks) > 3 and all(c.is_table for c in chunks)
    body = []
    for c in chunks:
        lines = c.text.splitlines()
        assert lines[:2] == ["Net sales by product line (in millions):", "Line | 2025 | 2024 | Change"]
        assert len(c.text.split()) <= 120
        body += lines[2:]
    assert body == rows  # every row exactly once, in order


def test_long_lead_in_paragraph_gives_only_its_last_sentence_as_caption():
    lead = PROSE * 2 + "Operating income by segment was as follows (in millions):"
    sec = _section(lead, "Segment | 2025 | 2024", "Robots | 310 | 280", "Services | 190 | 120")
    table = next(c for c in chunk_section(sec, HEADER) if c.is_table)
    assert table.text.splitlines()[0] == "Operating income by segment was as follows (in millions):"


def test_layout_tables_without_numbers_stay_prose():
    bullets = [
        "• | We depend on a single supplier for lidar sensors used in every robot we sell.",
        "• | Competition in warehouse automation is intense and larger rivals may cut prices.",
        "• | Export controls may restrict sales of our robots to certain countries.",
    ]
    chunks = chunk_section(_section(PROSE, *bullets), HEADER)
    assert len(chunks) == 1 and not chunks[0].is_table
    assert all(b in chunks[0].text for b in bullets)


def test_no_chunk_is_only_the_overlap_of_the_previous_one():
    rows = ["Segment | 2025 | 2024", "Robots | 1,700 | 1,480", "Services | 700 | 534"]
    sec = _section(PROSE * 3, *rows)
    chunks = chunk_section(sec, HEADER, target_words=200, overlap_words=30)
    assert chunks[-1].is_table
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))


def test_fixture_mdna_table_is_retrievable_with_its_caption(tenk_text):
    mdna = next(s for s in split_items(tenk_text) if s.item == "7")
    table = next(c for c in chunk_section(mdna, HEADER) if c.is_table)
    assert table.text.startswith("Total revenue grew 18% to $2.4 billion")  # the lead-in is the caption
    assert "Robots | 1,700 | 1,480 | 15" in table.text and "Services | 700 | 534 | 31" in table.text


def test_rechunk_replaces_a_filings_chunks_instead_of_skipping_or_doubling(db, embedder):
    from filings_rag.ingest import ingest_text
    from tests.conftest import make_filing

    filing = make_filing("RCHK", "Rechunk Corp", "0000000009-25-000001")
    text = "Item 1A. Risk Factors\n" + "Supplier risk could stop production. " * 60

    def stored():
        return db.execute(
            "SELECT c.id, c.text FROM chunks c JOIN filings f ON f.id = c.filing_id WHERE f.ticker = 'RCHK'"
        ).fetchall()

    n = ingest_text(db, embedder, filing, text, items=None)
    before = stored()
    assert n == len(before) > 0
    assert ingest_text(db, embedder, filing, text, items=None) == 0  # already indexed: skipped
    assert ingest_text(db, embedder, filing, text, items=None, replace=True) == n
    after = stored()
    assert len(after) == n and {r["id"] for r in after}.isdisjoint(r["id"] for r in before)
