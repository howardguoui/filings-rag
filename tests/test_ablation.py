"""Reranker ablation: the same questions with no rerank and with the reranker over each candidate pool size."""

import json
from pathlib import Path

import pytest

from evals import run_evals
from evals.run_evals import percentile, reranker_ablation, retrieval_metrics, write_report
from filings_rag.config import Settings
from filings_rag.rerank import KeywordOverlapReranker
from filings_rag.retrieval import Retriever
from tests.test_evals import _fixture_questions

REPO = Path(__file__).resolve().parent.parent


class PoolSpyReranker(KeywordOverlapReranker):
    """Deterministic keyword reranker that records how many candidates each call was handed."""

    def __init__(self):
        self.pools: list[int] = []

    def rerank(self, query, hits, k):
        self.pools.append(len(hits))
        return super().rerank(query, hits, k)


def test_ablation_runs_every_configuration_on_the_same_questions(db, embedder):
    spy = PoolSpyReranker()
    r = Retriever(db, embedder, spy, candidate_k=40)
    a = reranker_ablation(r, _fixture_questions(), k=3, candidates=(2, 4))

    assert a["n"] == 4 and a["k"] == 3 and a["reranker"] == "PoolSpyReranker"
    names = [c["config"] for c in a["configs"]]
    assert names == ["hybrid", "hybrid_rerank@2", "hybrid_rerank@4"]
    assert [c["candidates"] for c in a["configs"]] == [40, 2, 4]
    answerable = ["cyber", "lidar", "rates", "glbx-margin"]
    for c in a["configs"]:
        assert [p["id"] for p in c["per_question"]] == answerable, c["config"]
        assert 0 <= c["mrr"] <= c["hit_rate"] <= 1, c["config"]
        assert 0 <= c["median_ms"] <= c["p95_ms"], c["config"]

    no_rerank, at2, at4 = a["configs"]
    assert no_rerank["rerank"] is False and no_rerank["median_rerank_ms"] is None and "vs_no_rerank" not in no_rerank
    for c in (at2, at4):
        assert c["rerank"] is True and c["median_rerank_ms"] >= 0
        assert c["vs_no_rerank"]["hit_rate"] == pytest.approx(c["hit_rate"] - no_rerank["hit_rate"])
        assert c["vs_no_rerank"]["mrr"] == pytest.approx(c["mrr"] - no_rerank["mrr"])
        assert c["vs_no_rerank"]["median_ms"] == pytest.approx(c["median_ms"] - no_rerank["median_ms"])

    # Each rerank configuration ran its warm-up plus one search per question, with its own pool size
    assert len(spy.pools) == 2 * (1 + 4)
    assert max(spy.pools[:5]) <= 2 and max(spy.pools[5:]) <= 4 and max(spy.pools[5:]) > 2
    # A pool of 2 can only fill 2 of the 3 slots
    assert all(p["rank"] is None or p["rank"] <= 2 for p in at2["per_question"])
    # The production pool size is restored afterwards
    assert r.candidate_k == 40


def test_ablation_no_rerank_row_matches_the_hybrid_retrieval_row(db, embedder):
    r = Retriever(db, embedder, KeywordOverlapReranker())
    a = reranker_ablation(r, _fixture_questions(), k=3)
    hybrid = retrieval_metrics(r, _fixture_questions(), ["hybrid"], k=3)["hybrid"]
    assert a["configs"][0]["hit_rate"] == hybrid["hit_rate"] and a["configs"][0]["mrr"] == hybrid["mrr"]
    assert a["configs"][0]["per_question"] == hybrid["per_question"]


def test_ablation_needs_a_reranker(db, embedder):
    with pytest.raises(RuntimeError, match="RERANK_ENABLED"):
        reranker_ablation(Retriever(db, embedder, None), _fixture_questions(), k=3)


def test_percentile_is_nearest_rank():
    assert percentile([5.0], 95) == 5.0
    assert percentile([float(x) for x in range(1, 21)], 95) == 19.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0], 100) == 3.0


def test_full_run_with_ablation_writes_the_comparison_table(db, embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "RESULTS", tmp_path)
    monkeypatch.setattr(run_evals, "load_questions", _fixture_questions)
    r = Retriever(db, embedder, KeywordOverlapReranker(), candidate_k=40)
    result = run_evals.main(
        ["hybrid", "hybrid_rerank"], None, None, None, True, retriever=r, settings=Settings(top_k=3), ablation=True
    )
    assert [c["candidates"] for c in result["ablation"]["configs"]] == [40, 20, 40]

    saved = json.loads((tmp_path / "latest.json").read_text())
    assert saved["ablation"] == result["ablation"]
    stamped = json.loads((tmp_path / f"{run_evals.result_stamp(saved['run_at'])}.json").read_text())
    assert "ablation" in stamped
    md = (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert "## Reranker ablation (4 questions, hybrid search, top 3, reranker `KeywordOverlapReranker`)" in md
    assert "| no rerank | 40 |" in md and "| rerank top 20 | 20 |" in md and "| rerank top 40 | 40 |" in md


def test_run_without_the_flag_has_no_ablation(db, embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "RESULTS", tmp_path)
    monkeypatch.setattr(run_evals, "load_questions", _fixture_questions)
    r = Retriever(db, embedder, KeywordOverlapReranker())
    result = run_evals.main(["hybrid"], None, None, None, True, retriever=r, settings=Settings(top_k=3))
    assert "ablation" not in result
    assert "Reranker ablation" not in (tmp_path / "latest.md").read_text(encoding="utf-8")


def test_ablation_report_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "RESULTS", tmp_path)
    row = {"per_question": []}
    result = {
        "run_at": "2026-10-30 12:00 UTC",
        "n_questions": 42,
        "k": 6,
        "embed_model": "m",
        "retrieval": {"hybrid": {"hit_rate": 0.84, "mrr": 0.8, "median_ms": 10.0, "per_question": []}},
        "ablation": {
            "n": 42,
            "k": 6,
            "reranker": "Xenova/ms-marco-MiniLM-L-6-v2",
            "configs": [
                {**row, "config": "hybrid", "rerank": False, "candidates": 40, "hit_rate": 0.84, "mrr": 0.8,
                 "median_ms": 10.0, "p95_ms": 14.0, "median_rerank_ms": None},
                {**row, "config": "hybrid_rerank@20", "rerank": True, "candidates": 20, "hit_rate": 0.81,
                 "mrr": 0.78, "median_ms": 850.0, "p95_ms": 990.0, "median_rerank_ms": 838.0,
                 "vs_no_rerank": {"hit_rate": -0.03, "mrr": -0.02, "median_ms": 840.0}},
                {**row, "config": "hybrid_rerank@40", "rerank": True, "candidates": 40, "hit_rate": 0.9,
                 "mrr": 0.85, "median_ms": 1700.0, "p95_ms": 2100.0, "median_rerank_ms": 1650.0,
                 "vs_no_rerank": {"hit_rate": 0.06, "mrr": 0.05, "median_ms": 1690.0}},
            ],
        },
    }  # fmt: skip
    text = write_report(result).read_text(encoding="utf-8")
    assert "## Reranker ablation (42 questions, hybrid search, top 6, reranker `Xenova/ms-marco-MiniLM-L-6-v2`)" in text
    assert "| no rerank | 40 | 84% | 0.80 | 10 ms | 14 ms | – | – | – | – |" in text
    assert "| rerank top 20 | 20 | 81% | 0.78 | 850 ms | 990 ms | 838 ms | -3 pts | -0.02 | +840 ms |" in text
    assert "| rerank top 40 | 40 | 90% | 0.85 | 1700 ms | 2100 ms | 1650 ms | +6 pts | +0.05 | +1690 ms |" in text
    header = next(line for line in text.splitlines() if line.startswith("| Configuration |"))
    separator = text.splitlines()[text.splitlines().index(header) + 1]
    assert header.count("|") == separator.count("|")


def test_eval_command_passes_the_ablation_flags(monkeypatch):
    from filings_rag import cli

    calls = []
    monkeypatch.setattr(run_evals, "main", lambda *a, **kw: calls.append(kw))
    cli.main(["eval", "--skip-generation", "--ablation", "--ablation-candidates", "10", "30"])
    cli.main(["eval", "--skip-generation"])
    assert calls[0] == {"ablation": True, "ablation_candidates": (10, 30)}
    assert calls[1] == {"ablation": False, "ablation_candidates": (20, 40)}


def test_publish_script_runs_the_ablation():
    script = (REPO / "scripts" / "publish_results.sh").read_text(encoding="utf-8")
    eval_line = next(line for line in script.splitlines() if line.strip().startswith("filings-rag eval"))
    assert "--ablation" in eval_line


def test_ablation_report_never_prints_negative_zero():
    from evals.run_evals import ablation_lines

    config = {"per_question": [], "p95_ms": 2.0, "hit_rate": 1.0, "mrr": 0.5, "median_ms": 1.0}
    lines = ablation_lines(
        {
            "n": 1,
            "k": 3,
            "reranker": "x",
            "configs": [
                {**config, "config": "hybrid", "rerank": False, "candidates": 40, "median_rerank_ms": None},
                {
                    **config,
                    "config": "hybrid_rerank@20",
                    "rerank": True,
                    "candidates": 20,
                    "median_rerank_ms": 0.2,
                    "vs_no_rerank": {"hit_rate": -0.001, "mrr": -0.001, "median_ms": -0.3},
                },
            ],
        }
    )
    assert lines[-1].endswith("| +0 pts | +0.00 | +0 ms |")
