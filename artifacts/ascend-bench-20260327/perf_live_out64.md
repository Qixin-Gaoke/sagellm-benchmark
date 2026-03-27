# E2E Benchmark Report

## Summary
- Rows: 8
- Avg TTFT (ms): 7937.84
- Avg TBT (ms): 284.47
- Output Throughput (tok/s): 0.00
- Avg Per-Request Throughput (tok/s): 3.53

## Results

| Model | Scenario | Precision | Batch | TTFT(ms) | TBT(ms) | TPS | P95(ms) |
|-------|----------|-----------|-------|----------|---------|-----|---------|
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b1 | live | 1 | 38457.67 | 108.24 | 1.41 | 45277.03 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b1 | live | 1 | 7229.33 | 586.93 | 0.78 | 45379.53 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b2 | live | 2 | 3252.02 | 94.39 | 7.56 | 11913.74 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b2 | live | 2 | 298.83 | 176.93 | 5.37 | 11401.44 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b4 | live | 4 | 338.37 | 223.43 | 4.44 | 14415.52 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b4 | live | 4 | 519.47 | 314.76 | 3.14 | 20350.22 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_random_b8 | live | 8 | 6044.11 | 312.43 | 3.13 | 28492.69 |
| Qwen/Qwen2.5-0.5B-Instruct | vllm_sharegpt_b8 | live | 8 | 7362.89 | 458.62 | 2.45 | 40344.38 |
