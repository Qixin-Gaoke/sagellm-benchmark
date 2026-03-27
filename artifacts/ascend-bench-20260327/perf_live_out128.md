# E2E Benchmark Report

## Summary
- Rows: 8
- Avg TTFT (ms): 4642.66
- Avg TBT (ms): 235.17
- Output Throughput (tok/s): 0.00
- Avg Per-Request Throughput (tok/s): 5.14

## Results

| Model | Scenario | Precision | Batch | TTFT(ms) | TBT(ms) | TPS | P95(ms) |
|-------|----------|-----------|-------|----------|---------|-----|---------|
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b1 | live | 1 | 189.92 | 108.70 | 9.57 | 14647.22 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b1 | live | 1 | 219.38 | 125.54 | 7.92 | 16163.55 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b2 | live | 2 | 224.39 | 154.95 | 5.12 | 19499.88 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b2 | live | 2 | 317.68 | 204.13 | 4.90 | 26436.00 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b4 | live | 4 | 10046.12 | 171.22 | 4.79 | 38000.41 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b4 | live | 4 | 12434.69 | 242.67 | 3.74 | 52299.94 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b8 | live | 8 | 12428.95 | 365.81 | 2.77 | 65062.15 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b8 | live | 8 | 1280.16 | 508.35 | 2.31 | 73648.06 |
