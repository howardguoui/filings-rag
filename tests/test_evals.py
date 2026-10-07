import json

from evals import run_evals
from evals.run_evals import Question, is_hit, load_questions, retrieval_metrics, write_report
from filings_rag.config import Settings
from filings_rag.rerank import KeywordOverlapReranker
from filings_rag.retrieval import Retriever


def test_question_file_is_well_formed():
    qs = load_questions()
    ids = [q.id for q in qs]
    assert len(ids) == len(set(ids)) and len(qs) >= 30
    for q in qs:
        assert q.tickers, q.id
        assert q.unanswerable or q.sections, q.id
    assert sum(q.unanswerable for q in qs) >= 3
    assert any(not q.filter_tickers for q in qs)


def test_is_hit_matches_company_and_section():
    q = Question("x", "?", ["BAC"], ["1C"], True, False)
    assert is_hit(q, "bac", "1c")
    assert not is_hit(q, "JPM", "1C") and not is_hit(q, "BAC", "1A")


def _fixture_questions():
    return [
        Question("cyber", "Who oversees cybersecurity risk at the Audit Committee?", ["ACME"], ["1C"], True, False),
        Question("lidar", "Which supplier risk affects lidar sensors?", ["ACME"], ["1A"], True, False),
        Question(
            "rates", "What happens to interest expense if rates rise 100 basis points?", ["ACME"], ["7A"], True, False
        ),
        Question("glbx-margin", "Why did Globex Bank gross margin fall?", ["GLBX"], ["7"], False, False),
        Question("u", "Unanswerable?", ["ACME"], [], True, True),
    ]


def test_retrieval_metrics_on_fixture(db, embedder):
    r = Retriever(db, embedder, KeywordOverlapReranker())
    m = retrieval_metrics(r, _fixture_questions(), ["vector", "keyword", "hybrid", "hybrid_rerank"], k=3)
    assert set(m) == {"vector", "keyword", "hybrid", "hybrid_rerank"}
    for mode, v in m.items():
        assert 0 <= v["mrr"] <= v["hit_rate"] <= 1, mode
        assert len(v["per_question"]) == 4  # the unanswerable question is not scored for retrieval
    assert m["hybrid"]["hit_rate"] >= 0.75


def test_full_run_retrieval_only_writes_reports(db, embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "RESULTS", tmp_path)
    monkeypatch.setattr(run_evals, "load_questions", _fixture_questions)
    settings = Settings(embed_provider="hash", top_k=3)
    r = Retriever(db, embedder, None)
    result = run_evals.main(
        ["vector", "hybrid", "hybrid_rerank"], None, None, None, True, retriever=r, settings=settings
    )
    assert set(result["retrieval"]) == {"vector", "hybrid"}  # no reranker loaded -> mode skipped
    saved = json.loads((tmp_path / "latest.json").read_text())
    assert saved["n_questions"] == 4 and "generation" not in saved
    md = (tmp_path / "latest.md").read_text()
    assert "| hybrid |" in md and "| random ranking (baseline) |" in md
    assert 0 < saved["baseline"]["hit_rate"] < saved["retrieval"]["hybrid"]["hit_rate"]


def test_generation_metrics_with_fake_llm_and_no_judge(db, embedder):
    r = Retriever(db, embedder, None)
    g = run_evals.generation_metrics(r, _fixture_questions(), Settings(llm_provider="fake"), "fake", None, "hybrid", 3)
    assert g["model"] == "fake" and g["judge"] is None
    assert g["faithfulness"] is None  # no judge, no RAGAS scores
    assert g["citation_rate"] == 1.0
    assert len(g["rows"]) == 5 and "contexts" not in g["rows"][0]


def test_report_formats_generation_block(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "RESULTS", tmp_path)
    result = {
        "run_at": "2026-10-06 17:00 UTC",
        "n_questions": 2,
        "k": 6,
        "embed_model": "m",
        "retrieval": {"hybrid": {"hit_rate": 0.5, "mrr": 0.25, "median_ms": 12.0, "per_question": []}},
        "generation": {
            "model": "fake",
            "judge": "anthropic:claude",
            "faithfulness": 0.9,
            "answer_relevancy": None,
            "context_precision": 0.8,
            "citation_rate": 1.0,
            "abstention_rate": 1.0,
            "false_refusals": 0.0,
        },
    }
    text = write_report(result).read_text()
    assert "| faithfulness | 90% |" in text and "| answer relevancy | – |" in text


def test_expected_random_matches_simple_cases():
    from math import isclose

    from evals.run_evals import expected_random

    assert expected_random(10, 0, 5) == (0.0, 0.0)
    assert expected_random(4, 4, 1) == (1.0, 1.0)
    hit, mrr = expected_random(10, 1, 3)
    assert isclose(hit, 0.3) and isclose(mrr, (1 + 1 / 2 + 1 / 3) / 10)
    # brute force: 2 correct among 5, top 2
    from itertools import permutations

    ranks = []
    for perm in permutations(range(5)):
        first = next((i for i, x in enumerate(perm[:2], 1) if x < 2), None)
        ranks.append(first)
    bf_hit = sum(r is not None for r in ranks) / len(ranks)
    bf_mrr = sum(1 / r for r in ranks if r) / len(ranks)
    hit, mrr = expected_random(5, 2, 2)
    assert isclose(hit, bf_hit) and isclose(mrr, bf_mrr)


def test_result_file_names_are_digits_only():
    from evals.run_evals import result_stamp

    # The first real run was saved as "20261007 0011 U.json": the old slicing cut "UTC" in half.
    assert result_stamp("2026-10-07 00:11 UTC") == "202610070011"
