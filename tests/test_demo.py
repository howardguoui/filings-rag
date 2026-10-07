"""The static demo: recorded answers come from answer_question, and the built page runs without a server."""

import json

from filings_rag.cli import main
from filings_rag.demo import DEMO_QUESTIONS, STATIC_MARKER, build_site, record_answers
from filings_rag.llm import FakeLLM
from filings_rag.retrieval import Hit, RetrievalResult


class OneHitRetriever:
    reranker = None

    def __init__(self):
        self.seen = []

    def search(self, question, mode, k, filters):
        self.seen.append(filters.tickers)
        ticker = filters.tickers[0]
        hit = Hit(1, ticker, "Acme", 2025, "1A", "Risk Factors", "hdr", "Supplier risk text.", "https://x", 1.0)
        return RetrievalResult([hit], mode, {"search_ms": 3.0})


def test_record_answers_runs_every_question_with_its_company_filter():
    r = OneHitRetriever()
    rec = record_answers(r, FakeLLM(), mode="hybrid")
    assert len(rec["answers"]) == len(DEMO_QUESTIONS)
    assert r.seen == [t for _, t in DEMO_QUESTIONS]
    first = rec["answers"][0]
    assert first["question"] == DEMO_QUESTIONS[0][0] and first["tickers"] == ["JPM"]
    assert first["citations"][0]["section"] == "Item 1A. Risk Factors"
    assert "contexts" not in first  # the page needs citations and snippets, not whole chunks
    assert rec["model"] == FakeLLM().name and rec["mode"] == "hybrid" and rec["recorded_at"]


def test_build_site_writes_a_self_contained_static_page(tmp_path):
    rec = record_answers(OneHitRetriever(), FakeLLM(), questions=DEMO_QUESTIONS[:2])
    page = build_site(tmp_path, rec, {"run_at": "x", "retrieval": {}}, [{"ticker": "JPM", "company": "JPMorgan"}])
    html = page.read_text(encoding="utf-8")
    assert html.count(STATIC_MARKER) == 1
    assert 'href="static/app.css"' in html and 'src="static/app.js"' in html  # relative: works under /filings-rag/
    assert "/static/" not in html.replace("https://fonts.gstatic.com", "")
    assert (tmp_path / "static" / "app.js").exists() and (tmp_path / ".nojekyll").exists()
    data = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert data["filings"][0]["ticker"] == "JPM" and len(data["answers"]) == 2
    assert json.loads((tmp_path / "evals.json").read_text(encoding="utf-8"))["run_at"] == "x"
    build_site(tmp_path, rec, None, [])  # rebuilding keeps one marker
    assert page.read_text(encoding="utf-8").count(STATIC_MARKER) == 1


def test_page_script_switches_to_recorded_answers():
    from filings_rag.demo import STATIC_DIR

    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'meta[name="fr-mode"][content="static"]' in js
    assert "fetch('demo.json')" in js and "'evals.json'" in js


def test_demo_export_command_is_wired(monkeypatch):
    import filings_rag.cli as cli

    called = {}
    monkeypatch.setattr(cli, "cmd_demo_export", lambda args: called.update(vars(args)))
    main(["demo-export", "--llm", "fake", "--out", "site"])
    assert called["out"] == "site" and called["llm"] == "fake" and called["mode"] == "hybrid_rerank"
    assert called["evals"] == "evals/results/latest.json"
