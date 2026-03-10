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

- **Throughput**: Tokens generated per second
  - Higher is better
  - Important for batch processing

- **Error Rate**: Percentage of failed requests
  - Lower is better

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

# View raw JSON
sagellm-benchmark report --format json
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
