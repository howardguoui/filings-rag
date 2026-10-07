# Evaluation, 2026-10-07 00:11 UTC

32 questions · top 6 chunks · embeddings `BAAI/bge-small-en-v1.5`

## Retrieval

| Method | Hit rate | MRR | Median search |
| --- | --- | --- | --- |
| vector | 84% | 0.82 | 10 ms |
| keyword | 81% | 0.75 | 4 ms |
| hybrid | 84% | 0.83 | 12 ms |
| hybrid_rerank | 84% | 0.82 | 1762 ms |
| random ranking (baseline) | 63% | 0.39 | – |

## Answers (qwen3:8b, judged by ollama:qwen3:8b)

| Metric | Score |
| --- | --- |
| faithfulness | 92% |
| answer relevancy | 91% |
| context precision | 84% |
| citation rate | 81% |
| abstention rate | 100% |
| false refusals | 0% |
