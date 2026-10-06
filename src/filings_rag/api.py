"""FastAPI app: POST /api/ask returns a cited answer; the demo page is served at /."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .rag import answer_question
from .retrieval import Filters, Retriever

STATIC_DIR = Path(__file__).parent / "static"
RESULTS_DIR = Path("evals/results")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    tickers: list[str] | None = None
    fiscal_year: int | None = None
    mode: Literal["vector", "keyword", "hybrid", "hybrid_rerank"] | None = None
    k: int = Field(default=6, ge=1, le=10)


class RateLimiter:
    """Per-client sliding window plus a global daily cap, so a public demo can't run up an API bill."""

    def __init__(self, per_minute: int, daily_cap: int):
        self.per_minute = per_minute
        self.daily_cap = daily_cap
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.day = date.today()
        self.today = 0
        self.lock = threading.Lock()

    def check(self, client: str) -> None:
        now = time.monotonic()
        with self.lock:
            if date.today() != self.day:
                self.day, self.today = date.today(), 0
            if self.today >= self.daily_cap:
                raise HTTPException(429, "The demo's daily question limit is reached. Try again tomorrow.")
            q = self.hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(429, "Too many questions in a minute. Wait a moment and try again.")
            q.append(now)
            self.today += 1


def _client_id(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def create_app(settings: Settings | None = None, conn=None, embedder=None, llm=None, reranker=None) -> FastAPI:
    """Build the app. Tests inject conn/embedder/llm; production builds them from settings."""
    settings = settings or get_settings()
    state: dict = {"conn": conn, "embedder": embedder, "llm": llm, "reranker": reranker}
    limiter = RateLimiter(settings.rate_limit_per_minute, settings.daily_question_cap)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Load models and connect once at startup (anything not injected by tests).
        from .db import connect, init_schema
        from .embeddings import make_embedder
        from .llm import make_llm

        if state["conn"] is None:
            state["conn"] = connect(settings.database_url)
        if state["embedder"] is None:
            state["embedder"] = make_embedder(settings)
        init_schema(state["conn"], state["embedder"].dim)
        if state["llm"] is None:
            state["llm"] = make_llm(settings)
        if state["reranker"] is None and settings.rerank_enabled:
            from .rerank import CrossEncoderReranker

            state["reranker"] = CrossEncoderReranker(settings.rerank_model)
        yield

    app = FastAPI(title="filings-rag", version="0.1.0", lifespan=lifespan)

    def retriever() -> Retriever:
        return Retriever(state["conn"], state["embedder"], state["reranker"], settings.candidate_k)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "llm": getattr(state["llm"], "name", None),
            "reranker": state["reranker"] is not None,
            "embed_model": settings.embed_model if settings.embed_provider == "fastembed" else settings.embed_provider,
        }

    @app.get("/api/filings")
    def filings() -> list[dict]:
        from .db import list_filings

        rows = list_filings(state["conn"])
        return [{**r, "filing_date": str(r["filing_date"])} for r in rows]

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request) -> dict:
        if len(body.question) > settings.max_question_chars:
            raise HTTPException(422, f"Keep questions under {settings.max_question_chars} characters.")
        limiter.check(_client_id(request))
        try:
            ans = answer_question(
                body.question,
                retriever(),
                state["llm"],
                mode=body.mode or settings.retrieval_mode,
                k=body.k,
                filters=Filters(tickers=body.tickers, fiscal_year=body.fiscal_year),
                max_tokens=settings.max_answer_tokens,
            )
        except HTTPException:
            raise
        except Exception as exc:  # model/API failure: say what happened, don't leak a stack trace
            raise HTTPException(502, f"The answer model failed: {type(exc).__name__}") from exc
        out = asdict(ans)
        out.pop("contexts")
        return out

    @app.get("/api/evals/latest")
    def evals_latest() -> dict:
        path = RESULTS_DIR / "latest.json"
        if not path.exists():
            raise HTTPException(404, "No evaluation results yet. Run: filings-rag eval")
        return json.loads(path.read_text())

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def app_factory() -> FastAPI:  # uvicorn --factory filings_rag.api:app_factory
    return create_app()
