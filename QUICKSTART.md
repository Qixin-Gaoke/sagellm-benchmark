# sageLLM Benchmark - Quick Start Guide

## What is this?

sageLLM Benchmark is a standardized testing suite for validating LLM inference engines. It runs **Q1-Q8 query workloads** covering diverse LLM scenarios:

| Workload | Type | Prompt Tokens | Max Output | Requests | Concurrent |
|----------|------|---------------|------------|----------|------------|
| Q1 | Short Q&A | 32 | 64 | 5 | No |
| Q2 | Long context summarization | 512 | 128 | 3 | No |
| Q3 | Code generation | 128 | 256 | 3 | No |
| Q4 | Multi-turn conversation | 256 | 256 | 3 | No |
| Q5 | Concurrent short | 32 | 64 | 10 | Yes |
| Q6 | Concurrent long context | 512 | 256 | 10 | Yes |
| Q7 | Chain-of-thought reasoning | 256 | 512 | 3 | No |
| Q8 | Composite task | 192 | 128 | 4 | Yes |

## Installation

```bash
pip install isagellm-benchmark
```

如需安装第三方对比引擎客户端，统一通过 benchmark extras：

```bash
pip install -U 'isagellm-benchmark[vllm-client]'
pip install -U 'isagellm-benchmark[vllm-ascend-client]'
pip install -U 'isagellm-benchmark[lmdeploy-client]'
```

在 Ascend 机器上做 endpoint 对比时，如需复现已验证版本矩阵，再额外执行便利脚本：

```bash
sagellm-benchmark vllm-compare install-ascend
```

完整流程见 [docs/ASCEND_BENCHMARK.md](docs/ASCEND_BENCHMARK.md)。

标准 `sageLLM vs vLLM` live 对比便利入口：

```bash
sagellm-benchmark vllm-compare run \
  --sagellm-url http://127.0.0.1:8901/v1 \
  --vllm-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

对于 A100/CUDA 主机，推荐把 vLLM 固定放进 Docker 容器里运行，避免宿主机每次重复拉取 wheel、重配 torch ABI、或在冷启动时因为本地环境漂移导致失败：

```bash
cd sagellm-benchmark
VLLM_GPU_DEVICE=1 VLLM_PORT=9100 ./scripts/start_vllm_cuda_docker.sh

sagellm-benchmark vllm-compare run \
  --sagellm-url http://127.0.0.1:8901/v1 \
  --vllm-url http://127.0.0.1:9100/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct

# Sequential mode for single-GPU or tight-memory validation
sagellm-benchmark compare-record \
  --label sagellm \
  --url http://127.0.0.1:8901/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir ./benchmark_results/sequential/sagellm

sagellm-benchmark compare-record \
  --label vllm \
  --url http://127.0.0.1:9100/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir ./benchmark_results/sequential/vllm

sagellm-benchmark compare-offline \
  --result sagellm=./benchmark_results/sequential/sagellm/sagellm.json \
  --result vllm=./benchmark_results/sequential/vllm/vllm.json \
  --output-dir ./benchmark_results/sequential/compare
```

如需在 endpoint 缺失时由 benchmark 自动拉起 Dockerized vLLM：

```bash
sagellm-benchmark vllm-compare run \
  --sagellm-url http://127.0.0.1:8901/v1 \
  --vllm-url http://127.0.0.1:9100/v1 \
  --start-vllm-cmd "./scripts/start_vllm_cuda_docker.sh" \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

默认不会给 vLLM 容器加 `--rm`，这样如果启动失败可以直接检查日志：

```bash
docker logs sagellm-benchmark-vllm | tail -n 200
```

该 helper 默认使用 `--network host`。这在 A100 服务器上通常更稳，因为部分机器的 Docker bridge 对外不可达，容器会在下载模型元数据时直接报 `Network is unreachable`。

## 5-Minute Quick Start

### Option 1: One-Click Script (Recommended)

```bash
cd sagellm-benchmark
./run_benchmark.sh
```

That's it! Results will be in `./benchmark_results/`

To validate recent `sagellm-core` shared-stream convergence and `sagellm-backend` paged/native convergence against live endpoints:

```bash
cd sagellm-benchmark
./run_benchmark.sh --profile convergence \
  --target before=http://127.0.0.1:8901/v1 \
  --target after=http://127.0.0.1:8902/v1 \
  --log-file before=/tmp/sagellm-before.log \
  --log-file after=/tmp/sagellm-after.log \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

This profile writes:

- `comparison.json` and `comparison.md`
- `validation_summary.json` and `VALIDATION.md`
- `REPRODUCE.sh`
- `<label>_info.json`
- `<label>_metrics.prom`
- `<label>_log_probe.json` when `--log-file` is provided

### Option 2: Manual CLI

```bash
# Run all Q1-Q8 workloads
sagellm-benchmark run --workload all --backend cpu

# View results
sagellm-benchmark report
```

## What Gets Generated?

After running, you'll have:

```
benchmark_results/
├── benchmark_summary.json       # Overall summary
├── Q1_metrics.json              # Q1 workload metrics
├── Q2_metrics.json              # Q2 workload metrics
├── ...
├── Q8_metrics.json              # Q8 workload metrics
└── REPORT.md                    # Human-readable report
```

Convergence profile outputs look like:

```text
benchmark_results/convergence_YYYYMMDD_HHMMSS/
├── comparison.json
├── comparison.md
├── validation_summary.json
├── VALIDATION.md
├── REPRODUCE.sh
├── before.json
├── before.md
├── before_info.json
├── before_metrics.prom
├── before_log_probe.json
├── after.json
├── after.md
├── after_info.json
├── after_metrics.prom
└── after_log_probe.json
```

## Understanding the Output

### Terminal Output

```
Benchmark Results
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Workload ┃ Requests ┃ Errors ┃ Avg TTFT (ms) ┃ Throughput (tok/s) ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Q1       │        5 │      0 │         10.00 │             100.00 │
│ Q2       │        3 │      0 │         15.00 │              75.00 │
│ ...      │      ... │    ... │           ... │                ... │
│ Q8       │        4 │      0 │         12.00 │              90.00 │
└──────────┴──────────┴────────┴───────────────┴────────────────────┘
```

### Key Metrics Explained

- **TTFT (Time to First Token)**: How long until the first word appears
  - Lower is better
  - Critical for interactive chat

- **TBT (Time Between Tokens)**: Decode-step latency once the first token has arrived
  - Lower is better
  - Best field for comparing stateful batch decode convergence

- **Throughput**: Tokens generated per second
  - Higher is better
  - Important for batch processing

- **Output Throughput**: Output tokens per second over the full run
  - Higher is better
  - Best field for shared-stream and paged/native convergence comparison

- **Error Rate**: Percentage of failed requests
  - Lower is better

For convergence validation, also inspect the probe fields produced by `validation_summary.json`:

- `shared_stream_markers.hits`: log evidence that shared batching actually activated
- `paged_path_markers.hits`: log evidence that paged/native or fallback implementation paths were exercised
- `block_table_markers.hits`: log evidence that scheduler-provided block tables reached the runtime path

### How to map fields back to the mainline architecture

- If `avg_tbt_ms` improves but `block_table_markers.hits` and `paged_path_markers.hits` stay `0`, you only proved a latency change, not that paged KV or native attention mainline was exercised.
- If `shared_stream_markers.hits` is `0`, do not claim shared-stream convergence even if aggregate throughput improved.
- If marker hits are present but `avg_tbt_ms` and `output_throughput_tps` do not improve, the path is wired but not yet optimized.
- Strong convergence evidence means metric deltas and path evidence move together.

Canonical ownership and boundary rules are defined in <https://github.com/intellistream/sagellm-docs/blob/main/docs/specs/performance_mainline_architecture.md>.

## Backend Options

### CPU Backend

Real inference using HuggingFace models. Use for:
- Testing without GPU
- Validating correctness
- Getting realistic metrics

```bash
sagellm-benchmark run --backend cpu --model gpt2
```

**Note**: CPU inference is slow. Use small models like `gpt2` (124M params).

## Common Commands

```bash
# Run specific query workload
sagellm-benchmark run --workload Q3 --backend cpu

# Run in batch mode (offline throughput, vLLM/SGLang compatible)
sagellm-benchmark run --workload all --backend cpu --mode batch

# Run with custom JSON output
sagellm-benchmark run --workload all --backend cpu --output-json ./my_results.json

# Run with verbose logging
sagellm-benchmark run --workload all --backend cpu -v

# Custom output directory
sagellm-benchmark run --output ./my_results

# Generate markdown report
sagellm-benchmark report --format markdown > REPORT.md

# External convergence validation against two live endpoints
./run_benchmark.sh --profile convergence \
  --target baseline=http://127.0.0.1:8901/v1 \
  --target candidate=http://127.0.0.1:8902/v1 \
  --log-file baseline=/var/log/sagellm-baseline.log \
  --log-file candidate=/var/log/sagellm-candidate.log \
  --batch-size 1 --batch-size 2 --batch-size 4 \
  --model Qwen/Qwen2.5-0.5B-Instruct

# View raw JSON
sagellm-benchmark report --format json
```

Ascend-first startup example for a candidate endpoint:

```bash
cd /home/shuhao/sagellm
./scripts/sagellm_with_ascend_env.sh sagellm serve \
  --backend ascend \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --port 8902
```

## Troubleshooting

### Command not found

```bash
pip install isagellm-benchmark
```

### Need model for CPU backend

```bash
sagellm-benchmark run --backend cpu --model gpt2
```

### Results not found

Run benchmark first:
```bash
sagellm-benchmark run --workload all --backend cpu
```

## Next Steps

- Read [USAGE.md](../docs/USAGE.md) for detailed documentation
- Check [examples/](../examples/) for sample outputs
- See [README.md](../README.md) for architecture details
- **NEW:** See "Benchmarking Against vLLM/SGLang" section in [USAGE.md](../docs/USAGE.md) for comparison guide

## Comparing with vLLM/SGLang

sageLLM Benchmark now supports modes compatible with vLLM and SGLang:

```bash
# Batch mode - comparable to vLLM's offline throughput
sagellm-benchmark run --workload all --backend cpu --mode batch

# Traffic mode - comparable to SGLang's serving benchmark
sagellm-benchmark run --workload all --backend cpu --mode traffic
```

**Output includes vLLM/SGLang compatible metrics:**
- Request Throughput (req/s)
- Input Throughput (tokens/s)
- Output Throughput (tokens/s)  
- Total Throughput (tokens/s)

See [USAGE.md - Benchmarking Against vLLM/SGLang](../docs/USAGE.md#benchmarking-against-vllmsglang) for detailed comparison workflow.

## Need Help?

- GitHub Issues: https://github.com/intellistream/sagellm-benchmark/issues
- Documentation: [docs/](../docs/)

## Summary

```bash
# Quick start in 3 commands:
pip install isagellm-benchmark
sagellm-benchmark run --workload all --backend cpu
sagellm-benchmark report
```

That's all you need to get started!

For regression-style endpoint validation, switch to:

```bash
./run_benchmark.sh --profile convergence --target baseline=http://127.0.0.1:8901/v1 --target candidate=http://127.0.0.1:8902/v1
```
