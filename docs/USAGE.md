# sageLLM Benchmark Usage Guide

## Quick Start

### 1. Installation

```bash
pip install isagellm-benchmark
```

### 2. Run Benchmark

推荐优先直接使用 CLI 主链路：

```bash
# Canonical local workload benchmark path
sagellm-benchmark run --workload all --backend cpu

# Canonical live endpoint compare path
sagellm-benchmark compare \
  --target sagellm=http://127.0.0.1:8902/v1 \
  --target vllm=http://127.0.0.1:8901/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --hardware-family cuda
```

`run_benchmark.sh` 仍保留，但只作为兼容 shell wrapper：

```bash
./run_benchmark.sh
```

This will:
- Reuse `sagellm-benchmark run`
- Run the default Q1-Q8 local workload suite
- Generate metrics JSON files
- Generate canonical artifacts first
- Generate leaderboard exports only as compatibility artifacts

## CLI Reference

### `sagellm-benchmark compare`

统一比较多个 OpenAI-compatible endpoint。这是 sagellm vs vllm/lmdeploy 的正式 benchmark 入口。

**Options:**

- `--target`: 对比对象，格式 `LABEL=URL`
  - 示例：`sagellm=http://127.0.0.1:8902/v1`
  - 至少重复两次

- `--model`: 请求时使用的模型名

- `--batch-size`: 重复指定多个 batch 档位

- `--request-timeout`: 单请求超时秒数

- `--server-wait`: endpoint 判活最长等待秒数

- `--max-seq-len`: 覆盖自动探测的最大上下文窗口

- `--max-output-tokens`: 限制每请求输出长度

- `--output-dir`: 结果输出目录

**Examples:**

```bash
# Compare sagellm vs vllm
sagellm-benchmark compare \
  --target sagellm=http://127.0.0.1:8902/v1 \
  --target vllm=http://127.0.0.1:8901/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct

# Compare three engines with explicit batch sizes
sagellm-benchmark compare \
  --target sagellm=http://127.0.0.1:8902/v1 \
  --target vllm=http://127.0.0.1:8901/v1 \
  --target lmdeploy=http://127.0.0.1:23333/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 1 --batch-size 2 --batch-size 4 \
  --max-output-tokens 64
```

### `sagellm-benchmark run`

Canonical local workload benchmark pipeline.

**Options:**

- `--workload`: Workload selector
  - `all`: Full Q1-Q8 suite
  - `Q1` ... `Q8`: Single workload
  - `query`, `streaming`, `batch`, `mixed`: grouped selectors

-- `--backend`: Backend engine to use
  - `cpu`: CPU inference with HuggingFace (default)
  - `lmdeploy`: LMDeploy (coming soon)
  - `vllm`: vLLM (coming soon)

-- `--model`: Model path (for CPU backend)
  - Example: `sshleifer/tiny-gpt2`, `gpt2`

- `--output`, `-o`: Output directory
  - Default: `./benchmark_results`

- `--mode`: Benchmark mode (NEW - aligns with vLLM/SGLang)
  - `traffic`: Arrival pattern simulation (default)
  - `batch`: Offline batch throughput (all requests at once)

- `--output-json`: Path to save additional JSON output
  - Optional custom path for JSON results

- `--verbose`, `-v`: Enable verbose logging

**Examples:**

```bash
# Run the full Q1-Q8 suite with CPU backend (traffic mode)
sagellm-benchmark run --workload all --backend cpu

# Run in batch mode for offline throughput testing
sagellm-benchmark run --workload all --backend cpu --mode batch

# Run a single workload
sagellm-benchmark run --workload Q1 --backend cpu --model sshleifer/tiny-gpt2

# Custom output directory and JSON mirror
sagellm-benchmark run --workload Q4 --backend cpu -v -o ./my_results --output-json ./results.json
```

### `sagellm-benchmark report`

Generate helper reports from existing benchmark artifacts. This is not a benchmark execution mainline.

**Options:**

- `--input`, `-i`: Input summary JSON file
  - Default: `./benchmark_results/benchmark_summary.json`

- `--format`: Output format
  - `table`: Rich terminal table (default)
  - `json`: JSON output
  - `markdown`: Markdown format

**Examples:**

```bash
# Display table in terminal
sagellm-benchmark report

# Generate markdown report
sagellm-benchmark report --format markdown > REPORT.md

# Display raw JSON
sagellm-benchmark report --format json
```

## Workload Details

### Short Input (short_input)

- **Prompt tokens**: ~128
- **Output tokens**: 128
- **Requests**: 5 sequential
- **Purpose**: Test basic inference latency

### Long Input (long_input)

- **Prompt tokens**: ~200
- **Output tokens**: 200
- **Requests**: 3 sequential
- **Purpose**: Test long-context handling

### Stress Test (stress_test)

- **Prompt tokens**: ~256
- **Output tokens**: 256
- **Requests**: 10 concurrent
- **Purpose**: Test concurrent request handling and KV cache pressure

## Understanding Metrics

### Latency Metrics

- **TTFT (Time to First Token)**: Time from request submission to first token generation
  - Lower is better
  - Critical for interactive applications

- **TBT (Time Between Tokens)**: Average time between consecutive tokens
  - Lower is better
  - Affects user experience during streaming

- **TPOT (Time Per Output Token)**: Average time per generated token
  - Similar to TBT but includes first token
  - Lower is better

### Throughput Metrics

- **Throughput (tokens/sec)**: Number of tokens generated per second
  - Higher is better
  - Key metric for batch processing

#### vLLM/SGLang Compatible Throughput Metrics (NEW)

The benchmark now reports throughput metrics in a format compatible with vLLM and SGLang benchmarks:

- **Request Throughput (req/s)**: Number of requests processed per second
  - Formula: `successful_requests / total_time_s`
  - Comparable to vLLM's request throughput

- **Input Throughput (tokens/s)**: Input tokens processed per second
  - Formula: `total_input_tokens / total_time_s`
  - Measures prompt processing speed

- **Output Throughput (tokens/s)**: Output tokens generated per second
  - Formula: `total_output_tokens / total_time_s`
  - Measures generation speed

- **Total Throughput (tokens/s)**: Combined input + output tokens per second
  - Formula: `(total_input_tokens + total_output_tokens) / total_time_s`
  - Directly comparable to vLLM/SGLang total throughput

### Memory Metrics

- **Peak Memory (MB)**: Maximum memory usage during workload
  - Lower is better (for same performance)
  - Important for resource planning

- **Average Memory (MB)**: Average memory usage
  - Indicates typical resource consumption

### Reliability Metrics

- **Error Rate**: Proportion of failed requests
  - Lower is better
  - Target: < 5% for stress tests

- **Total/Successful/Failed Requests**: Request counts
  - Validate all requests are tracked

### KV Cache Metrics

- **KV Used Tokens**: Number of tokens in KV cache
  - Indicates cache utilization

- **KV Used Bytes**: Memory used by KV cache
  - Important for memory planning

- **Prefix Hit Rate**: Proportion of cache hits
  - Higher is better
  - Indicates cache efficiency

- **Evict Count**: Number of cache evictions
  - Lower is better (but expected under pressure)

- **Evict Time**: Time spent on evictions
  - Lower is better

### Compression Metrics

- **Spec Accept Rate**: Speculative decoding acceptance rate
  - Higher is better (when spec decoding is enabled)
  - 0.0 when not enabled

## Output Files

### Per-Workload Metrics

`{workload_name}_metrics.json`:

```json
{
  "avg_ttft_ms": 10.0,
  "p50_ttft_ms": 10.0,
  "p95_ttft_ms": 10.0,
  "p99_ttft_ms": 10.0,
  "avg_tbt_ms": 5.0,
  "avg_tpot_ms": 5.0,
  "avg_throughput_tps": 100.0,
  "peak_mem_mb": 1024,
  ...
}
```

### Summary Report

`benchmark_summary.json`:

```json
{
  "workloads": {
    "short_input": { ... },
    "long_input": { ... },
    "stress_test": { ... }
  },
  "overall": {
    "total_workloads": 3,
    "total_requests": 18,
    "successful_requests": 17,
    "failed_requests": 1
  }
}
```

## Backend Selection

### CPU Backend

**When to use:**
- Development without GPU
- Testing with real models
- Validating inference correctness
- Collecting realistic metrics

**Characteristics:**
- Real model loading (may be slow)
- Accurate TTFT/TBT/throughput
- Higher memory usage
- Slower than GPU

**Example:**
```bash
sagellm-benchmark run --workload m1 --backend cpu --model sshleifer/tiny-gpt2
```

## Troubleshooting

### "sagellm-benchmark not installed"

Install the package:
```bash
pip install isagellm-benchmark
```

### "Model path invalid"

CPU backend needs a valid model path:
```bash
sagellm-benchmark run --backend cpu --model sshleifer/tiny-gpt2
```

### "File not found: benchmark_summary.json"

Run benchmark first:
```bash
sagellm-benchmark run --workload m1 --backend cpu
```

### High memory usage with CPU backend

Use smaller models:
```bash
sagellm-benchmark run --backend cpu --model gpt2  # 124M params
# Or
sagellm-benchmark run --backend cpu --model facebook/opt-125m
```

### Slow execution with CPU backend

This is expected. CPU inference is much slower than GPU. Consider:
- Using smaller models
- Reducing num_requests in workload config

## Advanced Usage

### Custom Output Directory

```bash
sagellm-benchmark run --output ./my_results
sagellm-benchmark report --input ./my_results/benchmark_summary.json
```

### Verbose Logging

```bash
sagellm-benchmark run -v
```

### Running Single Workload

```bash
# Only short input
sagellm-benchmark run --workload short --backend cpu

# Only stress test
sagellm-benchmark run --workload stress --backend cpu --model sshleifer/tiny-gpt2
```

### Generating Reports

```bash
# Terminal table
sagellm-benchmark report

# Markdown for documentation
sagellm-benchmark report --format markdown > REPORT.md

# JSON for processing
sagellm-benchmark report --format json | jq '.overall'
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Benchmark

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install
        run: pip install isagellm-benchmark

      - name: Run Benchmark
        run: sagellm-benchmark run --workload m1 --backend cpu

      - name: Generate Report
        run: sagellm-benchmark report --format markdown > REPORT.md

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: |
            benchmark_results/
            REPORT.md
```

### GitLab CI Example

```yaml
benchmark:
  stage: test
  script:
    - pip install isagellm-benchmark
    - sagellm-benchmark run --workload year1 --backend cpu
    - sagellm-benchmark report --format markdown > REPORT.md
  artifacts:
    paths:
      - benchmark_results/
      - REPORT.md
```

## See Also

- [README.md](../README.md) - Overview and installation
- [examples/](../examples/) - Sample output files
---

## Benchmarking Against vLLM/SGLang

### Overview

sageLLM Benchmark can now produce results directly comparable to vLLM and SGLang benchmarks. This section explains how to run comparable tests.

### Batch Mode (Offline Throughput)

Both vLLM and SGLang offer "offline throughput" benchmarks where all requests are submitted at once and total time is measured. Use `--mode batch`:

```bash
# sageLLM batch mode (comparable to vLLM's throughput.py)
sagellm-benchmark run --workload m1 --backend cpu --mode batch --output-json batch_results.json

# vLLM equivalent (for reference)
# python benchmarks/throughput.py --model MODEL --num-prompts N --input-len X --output-len Y
```

**Key differences:**
- vLLM: Uses synthetic prompts with specified lengths
- sageLLM: Uses real prompts from ShareGPT or default dataset
- Both: Measure total throughput (tokens/s)

### Traffic Mode (Arrival Pattern Simulation)

For simulating realistic traffic with arrival patterns:

```bash
# sageLLM traffic mode (default)
sagellm-benchmark run --workload stress --backend cpu --mode traffic

# SGLang equivalent (for reference)
# python -m sglang.bench_serving --backend ... --request-rate R
```

### Comparing Metrics

When comparing results between frameworks, focus on these metrics:

| Metric | sageLLM Field | vLLM/SGLang Equivalent |
|--------|---------------|------------------------|
| Request throughput | `request_throughput_rps` | requests/s |
| Input throughput | `input_throughput_tps` | input tokens/s |
| Output throughput | `output_throughput_tps` | output tokens/s |
| Total throughput | `total_throughput_tps` | total tokens/s |
| Latency P50/P95/P99 | `p50_ttft_ms`, `p95_ttft_ms`, `p99_ttft_ms` | TTFT percentiles |

### Example Comparison Workflow

1. **Run sageLLM benchmark:**
```bash
sagellm-benchmark run --workload m1 --backend cpu --mode batch --output-json sagellm_results.json
```

2. **Extract comparable metrics:**
```bash
# View throughput metrics
sagellm-benchmark report --input ./benchmark_results/benchmark_summary.json

# Or parse JSON directly
cat sagellm_results.json | jq '.short_input | {
  request_throughput_rps,
  input_throughput_tps,
  output_throughput_tps,
  total_throughput_tps
}'
```

3. **Run vLLM benchmark (for comparison):**
```bash
# Example vLLM command (adapt to your setup)
python benchmarks/throughput.py \
  --model sshleifer/tiny-gpt2 \
  --num-prompts 100 \
  --input-len 128 \
  --output-len 128
```

4. **Compare results:**
   - Total throughput (tokens/s) should be directly comparable
   - Request throughput may differ due to different workload sizes
   - Input/output breakdown helps identify bottlenecks

### Best Practices for Fair Comparison

1. **Use similar hardware:** GPU type, memory, CPU cores
2. **Use same model:** Ensure identical model and quantization
3. **Match workload parameters:**
   - Similar prompt lengths
   - Similar output lengths  
   - Same number of requests
4. **Warm-up:** Both frameworks benefit from warm-up runs
5. **Multiple runs:** Average results from 3-5 runs for stability

### Limitations

- **Backend differences:** CPU vs GPU performance varies significantly
- **Implementation details:** Different frameworks may have different optimizations
- **Workload differences:** Real prompts (sageLLM) vs synthetic (vLLM) may show different patterns

### Getting Help

For questions about benchmark comparisons:
- GitHub Issues: https://github.com/intellistream/sagellm-benchmark/issues
- Check the vLLM/SGLang documentation for their benchmark details
- Review `THROUGHPUT_BENCHMARK_PLAN.md` for implementation details
