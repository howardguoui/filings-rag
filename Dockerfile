FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app

# Dependencies first (an empty package stands in for src), so a code change doesn't
# reinstall every wheel or re-download the models.
COPY pyproject.toml README.md ./
RUN mkdir -p src/filings_rag && touch src/filings_rag/__init__.py \
    && pip install . && pip uninstall -y filings-rag && rm -rf src

# Bake the embedding and reranker models into the image so cold starts don't download them.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY src ./src
RUN pip install --no-deps .

# Latest evaluation results, shown on the Evaluations tab
COPY evals/results ./evals/results

RUN useradd --create-home app && chown -R app /app /models
USER app

EXPOSE 8080
CMD ["sh", "-c", "uvicorn filings_rag.api:app_factory --factory --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
