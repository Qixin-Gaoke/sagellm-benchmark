# E2E Benchmark Report

## Summary
- Rows: 8
- Avg TTFT (ms): 1363.13
- Avg TBT (ms): 177.26
- Output Throughput (tok/s): 0.00
- Avg Per-Request Throughput (tok/s): 5.66

## Results

| Model | Scenario | Precision | Batch | TTFT(ms) | TBT(ms) | TPS | P95(ms) |
|-------|----------|-----------|-------|----------|---------|-----|---------|
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b1 | live | 1 | 1205.54 | 82.14 | 6.56 | 2437.70 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b1 | live | 1 | 212.50 | 105.32 | 8.91 | 1792.27 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b2 | live | 2 | 867.94 | 83.28 | 8.47 | 2746.38 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b2 | live | 2 | 1086.65 | 107.95 | 6.59 | 3492.59 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b4 | live | 4 | 1410.40 | 140.94 | 5.12 | 4184.25 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b4 | live | 4 | 1779.58 | 200.49 | 4.10 | 5748.93 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b8 | live | 8 | 1829.15 | 278.33 | 3.24 | 6641.78 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b8 | live | 8 | 2513.26 | 419.62 | 2.30 | 9786.28 |
