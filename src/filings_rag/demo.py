"""Record real answers to a fixed set of questions and build a static copy of the demo page.

The static copy (docs/ by default) runs on GitHub Pages with no server, database or API key: the page
shows the recorded answers and the latest evaluation instead of calling /api/ask. Every answer in it
comes from a real run of `filings-rag demo-export`; nothing is written by hand.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .llm import LLM
from .rag import answer_question
from .retrieval import Filters

STATIC_DIR = Path(__file__).parent / "static"
STATIC_MARKER = '<meta name="fr-mode" content="static" />'

# (question, company filter). The first four are the example buttons on the live page; the last one
# is unanswerable on purpose, to show the refusal.
DEMO_QUESTIONS: list[tuple[str, list[str]]] = [
    ("What cybersecurity risks does JPMorgan describe?", ["JPM"]),
    ("How did NVIDIA's data center revenue change, and why?", ["NVDA"]),
    ("What does Apple say about supply chain concentration?", ["AAPL"]),
    ("How sensitive is Bank of America's net interest income to rate changes?", ["BAC"]),
    ("What risks does Microsoft identify related to its development and use of AI?", ["MSFT"]),
    ("How do U.S. export controls affect NVIDIA's ability to sell products in China?", ["NVDA"]),
    ("What competitive risks does Amazon say it faces?", ["AMZN"]),
    ("How does Goldman Sachs describe its exposure to credit risk from counterparties?", ["GS"]),
    ("Where are Tesla's main manufacturing facilities located?", ["TSLA"]),
    ("How many cars did Microsoft sell last year?", ["MSFT"]),
]


def record_answers(
    retriever,
    llm: LLM,
    questions: list[tuple[str, list[str]]] = DEMO_QUESTIONS,
    mode: str = "hybrid_rerank",
    k: int = 6,
    max_tokens: int = 700,
) -> dict:
    """Answer each question for real and keep what the page shows (no full source texts)."""
    answers = []
    for question, tickers in questions:
        ans = answer_question(
            question, retriever, llm, mode=mode, k=k, filters=Filters(tickers=tickers), max_tokens=max_tokens
        )
        row = asdict(ans)
        row.pop("contexts", None)
        row["tickers"] = tickers
        answers.append(row)
    return {
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "model": llm.name,
        "mode": mode,
        "answers": answers,
    }


def build_site(out_dir: Path, recording: dict, evals: dict | None, filings: list[dict]) -> Path:
    """Write the page, its assets and the recorded data into out_dir (a GitHub Pages folder)."""
    out_dir = Path(out_dir)
    (out_dir / "static").mkdir(parents=True, exist_ok=True)
    for name in ("app.css", "app.js"):
        shutil.copyfile(STATIC_DIR / name, out_dir / "static" / name)
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    if STATIC_MARKER not in html:
        html = html.replace("<head>\n", "<head>\n" + STATIC_MARKER + "\n", 1)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")  # serve files as they are
    data = {**recording, "filings": filings}
    (out_dir / "demo.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    if evals is not None:
        (out_dir / "evals.json").write_text(json.dumps(evals, indent=2), encoding="utf-8")
    return out_dir / "index.html"
