# E2E Benchmark Report

## Summary
- Rows: 12
- Avg TTFT (ms): 57.39
- Avg TBT (ms): 11.56
- Avg Throughput (tok/s): 142.82

## Results

| Model | Scenario | Precision | Batch | TTFT(ms) | TBT(ms) | TPS | P95(ms) |
|-------|----------|-----------|-------|----------|---------|-----|---------|
| Qwen/Qwen2-7B-Instruct | vllm_random_b1 | fp16 | 1 | 44.41 | 8.33 | 102.60 | 1102.21 |
| Qwen/Qwen2-7B-Instruct | vllm_sharegpt_b1 | fp16 | 1 | 76.61 | 15.47 | 59.95 | 7980.05 |
| Qwen/Qwen2-7B-Instruct | vllm_random_b4 | fp16 | 4 | 48.13 | 8.82 | 186.28 | 448.24 |
| Qwen/Qwen2-7B-Instruct | vllm_sharegpt_b4 | fp16 | 4 | 83.31 | 15.43 | 101.64 | 2899.40 |
| Qwen/Qwen2-7B-Instruct | short_b8 | fp16 | 8 | 47.58 | 8.52 | 246.04 | 240.74 |
| Qwen/Qwen2-7B-Instruct | long_b8 | fp16 | 8 | 74.68 | 16.55 | 128.83 | 1584.44 |
| Qwen/Qwen2-7B-Instruct | vllm_random_b1 | int8 | 1 | 35.31 | 7.44 | 113.51 | 980.45 |
| Qwen/Qwen2-7B-Instruct | vllm_sharegpt_b1 | int8 | 1 | 65.12 | 14.78 | 66.83 | 7619.96 |
| Qwen/Qwen2-7B-Instruct | vllm_random_b4 | int8 | 4 | 38.81 | 8.21 | 190.29 | 411.34 |
| Qwen/Qwen2-7B-Instruct | vllm_sharegpt_b4 | int8 | 4 | 73.76 | 13.68 | 117.81 | 2570.48 |
| Qwen/Qwen2-7B-Instruct | short_b8 | int8 | 8 | 36.33 | 7.38 | 258.16 | 203.78 |
| Qwen/Qwen2-7B-Instruct | long_b8 | int8 | 8 | 64.67 | 14.13 | 141.85 | 1354.09 |
