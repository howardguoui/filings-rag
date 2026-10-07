# Evaluation, 2026-10-07 16:27 UTC

32 questions · top 6 chunks · embeddings `BAAI/bge-small-en-v1.5`

## Retrieval

| Method | Hit rate | MRR | Median search |
| --- | --- | --- | --- |
| vector | 84% | 0.82 | 8 ms |
| keyword | 81% | 0.75 | 3 ms |
| hybrid | 84% | 0.83 | 10 ms |
| hybrid_rerank | 84% | 0.82 | 1687 ms |
| random ranking (baseline) | 63% | 0.39 | – |

## Answers (qwen3:8b, judged by ollama:qwen3:8b)

| Metric | Score | Scored |
| --- | --- | --- |
| faithfulness | 97% | 30/32 |
| answer relevancy | 91% | 32/32 |
| context precision | 89% | 31/32 |
| citation rate | 100% |  |
| abstention rate | 100% |  |
| false refusals | 0% |  |
