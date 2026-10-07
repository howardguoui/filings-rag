"""Settings, read from environment variables or a .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "ollama", "vllm", "fake"]
EmbedProvider = Literal["fastembed", "ollama", "hash"]
RetrievalMode = Literal["vector", "keyword", "hybrid", "hybrid_rerank"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database (Postgres with the pgvector extension)
    database_url: str = "postgresql://postgres:postgres@localhost:5432/filings"

    # SEC EDGAR requires a User-Agent with a name and contact email.
    sec_user_agent: str = ""
    data_dir: str = "data"

    # Embeddings: the same model must be used for ingestion and for queries.
    embed_provider: EmbedProvider = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    ollama_embed_model: str = "nomic-embed-text"

    # Reranker (cross-encoder, runs on CPU through fastembed)
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_enabled: bool = True

    # Answer generation
    llm_provider: LLMProvider = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    max_answer_tokens: int = 700
    # Extra output room for local reasoning models (qwen3, deepseek-r1) on Ollama or vLLM: they think
    # before answering, and the thinking counts against max_tokens. Without it an answer can come back
    # empty because the budget ran out mid-thought.
    reasoning_tokens: int = 3072
    # Output limit for the RAGAS judge on Ollama or vLLM; RAGAS's own default is too small for a thinking judge.
    judge_max_tokens: int = 8192

    # Retrieval defaults
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    top_k: int = 6
    candidate_k: int = 40

    # Public-demo guard rails
    rate_limit_per_minute: int = 8
    daily_question_cap: int = 300
    max_question_chars: int = 500
    # Proxies in front of the app that append to X-Forwarded-For (Render's load balancer = 1).
    # The client address is read that many entries from the right, so a caller can't
    # dodge the rate limit by sending their own X-Forwarded-For.
    trusted_proxy_hops: int = 1
    db_pool_max: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
