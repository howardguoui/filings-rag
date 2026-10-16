"""Multi-company questions: find the companies named, search each one, interleave the rankings."""

import pytest

from evals.run_evals import Question, comparison_coverage, covers_every_company, retrieval_metrics, write_report
from filings_rag.query_plan import Company, interleave, mentioned_tickers, name_aliases, split_tickers
from filings_rag.rerank import KeywordOverlapReranker
from filings_rag.retrieval import Filters, Retriever

# Company names as EDGAR reports them for the companies the demo indexes
INDEX = [
    Company("AAPL", "Apple Inc."),
    Company("MSFT", "MICROSOFT CORP"),
    Company("NVDA", "NVIDIA CORP"),
    Company("AMZN", "AMAZON COM INC"),
    Company("JPM", "JPMORGAN CHASE & CO"),
    Company("BAC", "BANK OF AMERICA CORP /DE/"),
    Company("GS", "GOLDMAN SACHS GROUP INC"),
    Company("TSLA", "Tesla, Inc."),
]


@pytest.mark.parametrize(
    "company, aliases",
    [
        ("Apple Inc.", {"apple"}),
        ("AMAZON COM INC", {"amazon"}),
        ("BANK OF AMERICA CORP /DE/", {"bank of america"}),  # never just "bank"
        ("JPMORGAN CHASE & CO", {"jpmorgan chase", "jpmorgan"}),
        ("GOLDMAN SACHS GROUP INC", {"goldman sachs", "goldman"}),
        ("Tesla, Inc.", {"tesla"}),
        ("Inc.", set()),
    ],
)
def test_name_aliases_from_edgar_names(company, aliases):
    assert name_aliases(company) == aliases


def test_mentioned_tickers_in_order_of_first_mention():
    q = "Compare Microsoft's AI risks with Apple's, and with what JPMorgan says."
    assert mentioned_tickers(q, INDEX) == ["MSFT", "AAPL", "JPM"]
    assert mentioned_tickers("How do BAC and GS report value-at-risk?", INDEX) == ["BAC", "GS"]
    assert mentioned_tickers("how does bank of america govern cyber risk?", INDEX) == ["BAC"]


def test_tickers_must_be_capitalized_words():
    # "gs" inside a word, a lower-case "gs", and "bank" alone name no company
    assert mentioned_tickers("What do banks say about their earnings and gsm networks?", INDEX) == []
    assert mentioned_tickers("Which bank discloses the most?", INDEX) == []
    one_letter = [Company("A", "Agilent Technologies Inc."), Company("F", "FORD MOTOR CO")]
    assert mentioned_tickers("A question about Ford.", one_letter) == ["F"]


def test_split_tickers_rules():
    cmp = "Compare Apple and Microsoft on supply chain risk"
    assert split_tickers(cmp, INDEX, None) == ["AAPL", "MSFT"]
    assert split_tickers("What does Apple say about supply chain risk?", INDEX, None) == []
    assert split_tickers("What are the main risk factors?", INDEX, None) == []
    # A filter limits the split to the companies it allows
    assert split_tickers(cmp, INDEX, ["aapl"]) == []
    assert split_tickers(cmp + " and Tesla", INDEX, ["MSFT", "TSLA", "AAPL"]) == ["AAPL", "MSFT", "TSLA"]
    # Several companies picked, none named: compare across all of them
    assert split_tickers("Compare their cybersecurity governance.", INDEX, ["JPM", "BAC"]) == ["JPM", "BAC"]
    # Several picked, one named: just that company's question
    assert split_tickers("What does Apple say about China?", INDEX, ["AAPL", "MSFT"]) == []


def test_interleave_takes_turns_and_backfills():
    assert interleave([[1, 2, 3], ["a", "b", "c"]], 4) == [1, "a", 2, "b"]
    assert interleave([[1], ["a", "b", "c"]], 4) == [1, "a", "b", "c"]
    assert interleave([[], []], 4) == []


# Against the fixture database: ACME ("Acme Robotics Inc.") and GLBX ("Globex Bank Corp.")

CMP = "Compare the interest rate risk of Acme Robotics and Globex Bank."


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid", "hybrid_rerank"])
def test_comparison_retrieves_both_companies_in_turn(db, embedder, mode):
    res = Retriever(db, embedder, KeywordOverlapReranker()).search(CMP, mode=mode, k=4)
    assert res.split_tickers == ["ACME", "GLBX"]
    assert [h.ticker for h in res.hits] == ["ACME", "GLBX", "ACME", "GLBX"]


def test_split_embeds_once_and_sums_search_time(db, embedder):
    calls = []

    class Counting:
        dim = embedder.dim

        def embed_query(self, text):
            calls.append(text)
            return embedder.embed_query(text)

    res = Retriever(db, Counting()).search(CMP, mode="hybrid", k=4)
    assert calls == [CMP] and set(res.timings_ms) == {"embed_ms", "search_ms"}


def test_split_keeps_other_filters_and_can_be_turned_off(db, embedder):
    r = Retriever(db, embedder)
    res = r.search(CMP, mode="hybrid", k=6, filters=Filters(section_items=["7A"]))
    assert res.hits and all(h.section_item == "7A" for h in res.hits)
    assert {h.ticker for h in res.hits} == {"ACME", "GLBX"}
    off = r.search(CMP, mode="hybrid", k=4, split_companies=False)
    assert off.split_tickers == []


def test_single_company_questions_are_unchanged(db, embedder):
    r = Retriever(db, embedder)
    for q, f in (("What is Acme Robotics' interest rate risk?", None), ("revenue growth", Filters(tickers=["ACME"]))):
        a, b = (
            r.search(q, mode="hybrid", k=4, filters=f),
            r.search(q, mode="hybrid", k=4, filters=f, split_companies=False),
        )
        assert a.split_tickers == [] and [h.id for h in a.hits] == [h.id for h in b.hits]


def test_two_companies_picked_and_none_named_compares_them(db, embedder):
    res = Retriever(db, embedder).search(
        "Compare their cybersecurity governance.", mode="hybrid", k=4, filters=Filters(tickers=["GLBX", "ACME"])
    )
    assert res.split_tickers == ["GLBX", "ACME"] and [h.ticker for h in res.hits[:2]] == ["GLBX", "ACME"]


def test_covers_every_company():
    q = Question("c", "?", ["ACME", "GLBX"], ["7A"], False, False)

    class H:
        def __init__(self, ticker, item):
            self.ticker, self.section_item = ticker, item

    assert covers_every_company(q, [H("ACME", "7A"), H("GLBX", "7A")])
    assert not covers_every_company(q, [H("ACME", "7A"), H("ACME", "7A"), H("GLBX", "1A")])


def test_eval_reports_comparison_coverage(db, embedder, tmp_path, monkeypatch):
    from evals import run_evals

    qs = [
        Question("cmp", CMP, ["ACME", "GLBX"], ["7A"], False, False),
        Question("one", "Who oversees cybersecurity risk?", ["ACME"], ["1C"], True, False),
    ]
    r = Retriever(db, embedder)
    split = comparison_coverage(r, qs[:1], "hybrid", 4, split=True)
    assert split == {"rate": 1.0, "per_question": [{"id": "cmp", "every_company": True}]}

    m = retrieval_metrics(r, qs, ["hybrid"], k=4)
    c = m["hybrid"]["comparisons"]
    assert c["n"] == 1 and c["split"]["rate"] == 1.0 and 0 <= c["single_search"]["rate"] <= 1
    assert "comparisons" not in retrieval_metrics(r, qs[1:], ["hybrid"], k=4)["hybrid"]

    monkeypatch.setattr(run_evals, "RESULTS", tmp_path)
    write_report({"run_at": "2026-10-16 09:00 UTC", "n_questions": 2, "k": 4, "embed_model": "hash", "retrieval": m})
    md = (tmp_path / "latest.md").read_text()
    assert "## Comparison questions (1): every company named has its section in the top 4" in md
    assert "| hybrid | 100% |" in md
