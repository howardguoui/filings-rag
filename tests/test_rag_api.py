from fastapi.testclient import TestClient

from filings_rag.api import RateLimiter, create_app
from filings_rag.config import Settings
from filings_rag.llm import FakeLLM, Generation
from filings_rag.rag import answer_question, build_prompt, cited_numbers
from filings_rag.rerank import KeywordOverlapReranker
from filings_rag.retrieval import Filters, Hit, Retriever


def _hit(n: int) -> Hit:
    return Hit(n, "ACME", "Acme", 2025, "7", "MD&A", "hdr", f"Revenue fact number {n}.", "https://x", 1.0)


def test_cited_numbers_ignores_out_of_range_and_duplicates():
    assert cited_numbers("A [2]. B [1][2]. C [9]. D [0].", 3) == [2, 1]


def test_build_prompt_numbers_sources():
    p = build_prompt("Q?", [_hit(1), _hit(2)])
    assert "[1] Acme (ACME) 10-K FY2025, Item 7. MD&A" in p and "[2]" in p and p.endswith("Question: Q?")


def test_answer_with_fake_llm_cites_sources(db, embedder):
    r = Retriever(db, embedder, KeywordOverlapReranker())
    ans = answer_question(
        "How much did revenue grow?", r, FakeLLM(), mode="hybrid", k=4, filters=Filters(tickers=["ACME"])
    )
    assert ans.citations and ans.citations[0].n == 1
    assert ans.citations[0].ticker == "ACME"
    assert not ans.abstained
    assert "generate_ms" in ans.timings_ms and len(ans.contexts) == 4


def test_no_hits_means_abstain_without_calling_the_llm(db, embedder):
    class Boom:
        name = "boom"

        def generate(self, *a, **k):
            raise AssertionError("LLM must not be called")

    ans = answer_question("zzzqqqxx", Retriever(db, embedder), Boom(), mode="keyword")
    assert ans.abstained and ans.citations == []


def test_abstention_is_detected():
    class NoAnswer:
        name = "x"

        def generate(self, system, user, max_tokens):
            return Generation("The filings provided do not contain this information.", "x", 1, 1, 1.0)

    class OneHit:
        reranker = None

        def search(self, *a, **k):
            from filings_rag.retrieval import RetrievalResult

            return RetrievalResult([_hit(1)], "hybrid", {})

    ans = answer_question("q", OneHit(), NoAnswer())
    assert ans.abstained and ans.citations == []


def _client(db, embedder, **overrides) -> TestClient:
    s = Settings(rerank_enabled=False, rate_limit_per_minute=3, daily_question_cap=100, **overrides)
    return TestClient(create_app(s, conn=db, embedder=embedder, llm=FakeLLM(), reranker=KeywordOverlapReranker()))


def test_api_ask_returns_answer_and_citations(db, embedder):
    with _client(db, embedder) as c:
        r = c.post("/api/ask", json={"question": "What drove gross margin?", "tickers": ["ACME"]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["citations"][0]["section"].startswith("Item ")
        assert "contexts" not in body
        assert body["mode"] == "hybrid_rerank"


def test_api_validates_input(db, embedder):
    with _client(db, embedder) as c:
        assert c.post("/api/ask", json={"question": "x" * 600}).status_code == 422
        assert c.post("/api/ask", json={"question": "ok?", "k": 50}).status_code == 422
        assert c.post("/api/ask", json={"question": "ok?", "mode": "magic"}).status_code == 422


def test_api_rate_limits_per_client(db, embedder):
    with _client(db, embedder) as c:
        codes = [c.post("/api/ask", json={"question": "revenue?"}).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        other = c.post("/api/ask", json={"question": "revenue?"}, headers={"x-forwarded-for": "9.9.9.9"})
        assert other.status_code == 200


def test_daily_cap():
    lim = RateLimiter(per_minute=100, daily_cap=2)
    lim.check("a")
    lim.check("b")
    try:
        lim.check("c")
        raise AssertionError("expected 429")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 429


def test_api_filings_health_page_and_missing_evals(db, embedder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with _client(db, embedder) as c:
        assert {f["ticker"] for f in c.get("/api/filings").json()} == {"ACME", "GLBX"}
        assert c.get("/api/health").json()["ok"] is True
        assert "Ask the 10-K filings" in c.get("/").text
        assert c.get("/static/app.js").status_code == 200
        assert c.get("/api/evals/latest").status_code == 404


def test_api_reports_model_failures_as_502(db, embedder):
    class Broken:
        name = "broken"

        def generate(self, *a, **k):
            raise TimeoutError("model down")

    s = Settings(rerank_enabled=False)
    with TestClient(create_app(s, conn=db, embedder=embedder, llm=Broken(), reranker=None)) as c:
        r = c.post("/api/ask", json={"question": "revenue growth?"})
        assert r.status_code == 502 and "TimeoutError" in r.json()["detail"]
