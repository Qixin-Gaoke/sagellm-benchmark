# sagellm-benchmark

## Protocol Compliance (Mandatory)

- MUST follow Protocol v0.1: https://github.com/intellistream/sagellm-docs/blob/main/docs/specs/protocol_v0.1.md
- Any globally shared definitions (fields, error codes, metrics, IDs, schemas) MUST be added to Protocol first.

[![CI](https://github.com/intellistream/sagellm-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/intellistream/sagellm-benchmark/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/intellistream/sagellm-benchmark/branch/main/graph/badge.svg)](https://codecov.io/gh/intellistream/sagellm-benchmark)
[![PyPI version](https://badge.fury.io/py/isagellm-benchmark.svg)](https://badge.fury.io/py/isagellm-benchmark)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Private](https://img.shields.io/badge/License-Private-red.svg)](LICENSE)

Benchmark suite for sageLLM inference engine performance and validation.

New here? See [QUICKSTART.md](QUICKSTART.md) for a 5-minute guide.

Canonical boundary note: performance-path ownership is defined in <https://github.com/intellistream/sagellm-docs/blob/main/docs/specs/performance_mainline_architecture.md>. `sagellm-benchmark` validates the main path externally; it does not define runtime semantics for `sagellm-core`.

## Mainline Validation Mapping

Use benchmark output as proof for the canonical performance mainline, not as a standalone score table.

- `avg_ttft_ms`: admission + prefill responsiveness. Good for spotting whether batching changes hurt interactive startup.
- `avg_tbt_ms`: decode-step latency on the formal execution path. This is the first field to inspect for `stateful batch decode` convergence.
- `avg_throughput_tps`: average per-request throughput. Useful for request-level decode efficiency, but not a substitute for aggregate batch throughput.
- `output_throughput_tps`: best top-line field for shared-stream and paged/native convergence when comparing batch size `>= 2`.
- `request_throughput_rps`: useful when an optimization changes concurrent admission or batch turnover more than single-request token speed.
- `shared_stream_markers.hits`: required log evidence that shared batching actually activated.
- `paged_path_markers.hits`: evidence that paged/native or explicit fallback attention implementations were actually reached.
- `block_table_markers.hits`: evidence that scheduler-provided `block_tables` crossed the runtime boundary and survived to execution.

Interpretation rule:

- Better `avg_tbt_ms` or `output_throughput_tps` without marker evidence is only a performance observation, not proof that the intended mainline path converged.
- Marker evidence without competitive latency/throughput means the path is wired, but not yet performant.
- A valid convergence claim should combine metric deltas with `/info`, `/metrics`, or `*_log_probe.json` evidence.

## CUDA Decode Parity Gate

Round 0 的统一 gate 不再用“至少齐平”这类模糊说法，而是固定成一个可执行 schema：

- Workload: `short` (`128 -> 128`) + `long` (`2048 -> 512`)
- Batch sizes: `1 / 2 / 4`
- Warmup: `3`
- Measured rounds: `10`
- Pass rule: 每个 scenario 都必须同时满足
   - `correctness_pass_rate == 1.0`
   - `fallback_rate == 0.0`
   - 存在 step-level evidence
   - `candidate_tbt <= best_reference_tbt * 1.05`
   - `candidate_output_throughput >= best_reference_output_throughput * 0.95`

评估语义保持单一 gate，不再拆第二套 schema：

- `telemetry`、`performance`、`correctness`、`fallback`、`capability` 会分别落盘，并且同一个 scenario 可以同时出现多条失败事实。
- 缺少 step telemetry 不会再短路性能判断；如果旧工件同时 telemetry 缺失且性能退化，输出里会同时看到两类失败。
- 旧 `compare-record` / `compare` 的 `e2e` 工件仍然可直接评估，但会被诚实地标记为“缺少 fallback evidence”，而不是把 `fallback_rate` 伪造成 `0.0`。
- legacy `e2e` 工件的吞吐会优先读取 `output_throughput_tps`，否则退回现有 `throughput_tps` 作为兼容代理值；不会再额外乘 `batch_size`。

默认 gate 定义可直接导出：

```bash
sagellm-benchmark parity-gate print-default \
   --output ./benchmark_results/cuda_decode_parity_gate.json
```

用现有 `compare-record` / `compare` 工件做 gate 评估：

```bash
sagellm-benchmark parity-gate evaluate \
   --candidate ./benchmark_results/sagellm.json \
   --reference ./benchmark_results/vllm.json \
   --reference ./benchmark_results/sglang.json \
   --output ./benchmark_results/parity_evaluation.json
```

如果输入只是当前 `e2e` compare 工件而没有额外 step telemetry / fallback evidence，评估会明确报 `telemetry` 与 `fallback` 失败；若同时存在性能退化，也会一起报出，而不是被前面的证据缺口短路掉。

## Features

- Profile-first benchmark mainline (`vllm_random` / `vllm_sharegpt` / `vllm_hf` / `vllm_custom`)
- Q1-Q8 supplement workloads (`q1q8_supplement`) integrated into the same compare/run pipeline
- Standardized JSON metrics and reports
- One-command benchmark runner
- Extensible backend support
- Performance benchmark CLI (`perf`) for operator baselines and single-endpoint E2E sampling
- Canonical `compare` entrypoint for cross-engine live endpoint benchmarking
- Convergence validation profile for shared-stream batching, block-table usage, and paged/native path evidence

## Dependencies

- **isagellm-protocol** (>=0.4.0.0)
- **isagellm-core** (>=0.4.0.0)
- **isagellm-backend** (>=0.4.0.1)

## Installation

```bash
pip install isagellm-benchmark
```

For specific backend support:

```bash
# With vLLM support (non-Ascend)
pip install 'isagellm-benchmark[vllm-client]'

# With vLLM Ascend support (Ascend machines)
pip install 'isagellm-benchmark[vllm-ascend-client]'

# With LMDeploy support
pip install 'isagellm-benchmark[lmdeploy-client]'
```

Dependency policy:

- `pyproject.toml` extras are the single source of truth for third-party compare clients.
- `quickstart.sh` and setup scripts are convenience layers that install those extras and, when needed, add a validated runtime matrix on top.
- Cross-engine comparison belongs on the benchmark side only; `sagellm-core` is not a third-party engine compare entry.

## Quick Start

Recommended benchmark mainline:

- `sagellm-benchmark run` for local workload benchmarks.
- `sagellm-benchmark perf --live` for single-endpoint operator/e2e collection only.
- `sagellm-benchmark compare` as the only recommended cross-engine live benchmark entrypoint.
- `sagellm-benchmark vllm-compare install-ascend` only as an environment/bootstrap helper for validated Ascend compare setup.
- `./run_benchmark.sh` only as a compatibility shell wrapper.
- `sagellm-benchmark report` only as a reporting helper over existing artifacts.

```bash
# Mainline-only local run
sagellm-benchmark run --profile vllm_random --backend cpu --output ./benchmark_results

# Mainline + Q1-Q8 supplement
sagellm-benchmark run --profile vllm_random --supplement q1q8_supplement --backend cpu

# Generate a markdown report from existing artifacts
sagellm-benchmark report --input ./benchmark_results/benchmark_summary.json --format markdown

# Run migrated performance benchmarks
sagellm-benchmark perf --type operator --device cpu
sagellm-benchmark perf --type e2e --model Qwen/Qwen2-7B-Instruct --batch-size 1 --batch-size 4

# Live perf stays on a single endpoint; use compare for engine-vs-engine runs
sagellm-benchmark perf --type e2e --live \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --backend-url http://127.0.0.1:8901/v1

# Compare multiple OpenAI-compatible endpoints through benchmark clients
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8902/v1 \
   --target vllm=http://127.0.0.1:8901/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-0.5B-Instruct

# compare completion now also materializes a stable publish-ready snapshot directory at:
# <output-dir>/publish/website-ready/
# with leaderboard_single.json, leaderboard_multi.json,
# leaderboard_compare.json, and last_updated.json.

# Profile-first streaming benchmark over OpenAI-compatible endpoints
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8902/v1 \
   --target vllm=http://127.0.0.1:8901/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --profile vllm_random

# Mainline + supplement in one run
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8902/v1 \
   --target vllm=http://127.0.0.1:8901/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --profile vllm_random \
   --supplement q1q8_supplement

# Add --publish only when you also want the explicit HF dataset publish step.
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8902/v1 \
   --target vllm=http://127.0.0.1:8901/v1 \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --hardware-family cuda \
   --publish \
   --publish-hf-dataset intellistream/sagellm-benchmark-results

# compare already writes those four snapshots locally under publish/website-ready/.
# --publish only accepts the standard export boundary under the compare output,
# regenerates the same publish-ready snapshots locally, and uploads them to HF.

# Publish an existing compare output directory directly.
sagellm-benchmark publish \
   --input ./benchmark_results/compare \
   --hf-dataset intellistream/sagellm-benchmark-results

# Compatibility helpers for constrained environments: sequential capture, then offline summary.
sagellm-benchmark compare-record \
   --label sagellm \
   --url http://127.0.0.1:8901/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-1.5B-Instruct \
   --output-dir ./benchmark_results/sequential/sagellm

sagellm-benchmark compare-record \
   --label vllm \
   --url http://127.0.0.1:9100/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-1.5B-Instruct \
   --output-dir ./benchmark_results/sequential/vllm

# Dataset-backed sequential capture for memory-constrained hosts
sagellm-benchmark compare-record \
   --label sagellm \
   --url http://127.0.0.1:8901/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-1.5B-Instruct \
   --profile vllm_sharegpt \
   --num-prompts 64 \
   --input-len 1024 \
   --output-len 256 \
   --output-dir ./benchmark_results/sequential/sagellm

sagellm-benchmark compare-offline \
   --result sagellm=./benchmark_results/sequential/sagellm/sagellm.json \
   --result vllm=./benchmark_results/sequential/vllm/vllm.json \
   --output-dir ./benchmark_results/sequential/compare

# compare / compare-record will also best-effort capture runtime metadata from
# /info into <label>_info.json, and when the endpoint is sagellm-core with
# explicit decode telemetry enabled, emit <label>_core_telemetry.json.

# In an interactive terminal, compare can also prompt to kill local target
# processes after the benchmark finishes.
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8902/v1 \
   --target vllm=http://127.0.0.1:8901/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --prompt-cleanup

# If local endpoints are not running yet, provide per-target start commands.
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8902/v1 \
   --target vllm=http://127.0.0.1:8000/v1 \
   --target-command "sagellm=sagellm serve --backend cuda --model Qwen/Qwen2.5-0.5B-Instruct --port 8902" \
   --target-command "vllm=vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000" \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --prompt-cleanup

# Standard sageLLM vs vLLM compare on the canonical mainline
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8901/v1 \
   --target vllm=http://127.0.0.1:8000/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-0.5B-Instruct

# Prompt to clean up the locally running SageLLM/vLLM endpoints afterwards.
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8901/v1 \
   --target vllm=http://127.0.0.1:8000/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --prompt-cleanup

# Optionally auto-start local SageLLM/vLLM endpoints if they are not up yet.
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8901/v1 \
   --target vllm=http://127.0.0.1:8000/v1 \
   --target-command "sagellm=sagellm serve --backend cuda --model Qwen/Qwen2.5-0.5B-Instruct --port 8901" \
   --target-command "vllm=vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000" \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --prompt-cleanup

# Recommended on A100/CUDA hosts: keep vLLM in a dedicated Docker container,
# then reuse the stable endpoint for each compare run.
VLLM_GPU_DEVICE=1 VLLM_PORT=9100 \
   ./scripts/start_vllm_cuda_docker.sh

sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8901/v1 \
   --target vllm=http://127.0.0.1:9100/v1 \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-1.5B-Instruct

# Or let compare bootstrap the Dockerized vLLM endpoint on demand.
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8901/v1 \
   --target vllm=http://127.0.0.1:9100/v1 \
   --target-command "vllm=./scripts/start_vllm_cuda_docker.sh" \
   --hardware-family cuda \
   --model Qwen/Qwen2.5-1.5B-Instruct

# If startup fails, logs remain available because the helper does not use
# --rm by default:
docker logs sagellm-benchmark-vllm | tail -n 200

# The helper defaults to --network host, which is more reliable on locked-down
# servers where Docker bridge networking cannot reach huggingface.co.

# Generate charts (PNG/PDF, dark theme)
sagellm-benchmark perf --type e2e --plot --plot-format png --plot-format pdf --theme dark

# Compatibility shell wrapper over `sagellm-benchmark run`
./run_benchmark.sh

# Explicit publish dry-run from the benchmark CLI main path
sagellm-benchmark run \
   --profile vllm_random --supplement q1q8_supplement \
   --backend cpu \
   --model sshleifer/tiny-gpt2 \
   --output ./benchmark_results/publish_demo \
   --publish \
   --publish-dry-run

# Re-run publish later for an existing artifact directory
sagellm-benchmark publish \
   --input ./benchmark_results/publish_demo \
   --dry-run

# Compatibility shell wrapper over `sagellm-benchmark compare` plus extra probes
./run_benchmark.sh --profile convergence \
   --target before=http://127.0.0.1:8901/v1 \
   --target after=http://127.0.0.1:8902/v1 \
   --log-file before=/tmp/sagellm-before.log \
   --log-file after=/tmp/sagellm-after.log \
   --model Qwen/Qwen2.5-0.5B-Instruct
```

## Streaming Serving Benchmark

The canonical serving benchmark path is the stream-based `compare` pipeline. It talks to OpenAI-compatible `/v1/chat/completions` endpoints with `stream=true`, then aggregates request-level streaming metrics including TTFT, ITL, TPOT, E2EL, input throughput, output throughput, and request throughput.

`/v1/completions` with `stream=true` is not part of the benchmark support boundary. The benchmark client fails fast and only accepts `/v1/chat/completions` for streaming serving runs.

- Use `compare` for all real cross-engine comparisons.
- Use `compare-record` plus `compare-offline` when endpoints cannot stay up at the same time.
- `perf --live` remains a single-endpoint sampling path only; it is not the recommended engine-vs-engine benchmark entry.

## Canonical Publish Chain

The productized compare path is now:

1. `compare` captures streaming endpoint results and writes `*.canonical.json`.
2. `export_standard_leaderboard_artifacts()` derives `*_leaderboard.json` plus `leaderboard_manifest.json`.
3. The compare command itself writes a stable local publish-ready directory at `<compare-output>/publish/website-ready/` with:
   - `leaderboard_single.json`
   - `leaderboard_multi.json`
   - `leaderboard_compare.json`
   - `last_updated.json`
4. `publish` reuses that same output boundary, uploads canonical per-entry JSON into the HF dataset, and optionally syncs the snapshot files into `sagellm-website/data/`.
5. website consumes only those standard snapshots from HF or from a locally synced compatibility cache.

`leaderboard_compare.json` is a derived website artifact, not a second benchmark result schema. It is generated strictly from validated leaderboard entries so the website can directly render `sageLLM vs vLLM` deltas without hand-editing JSON.

Prompt source options for the stream compare pipeline:

- `--profile vllm_random`: synthetic prompts sampled through the serving dataset builder.
- `--profile vllm_sharegpt`: ShareGPT-backed prompts sampled through the same serving dataset builder.
- `--profile vllm_hf`: HuggingFace-backed ShareGPT sampling profile.
- `--profile vllm_custom`: user-provided dataset path + explicit prompt shape.

`vllm_custom` requires the full tuple `--dataset-path`, `--num-prompts`, `--input-len`, and `--output-len` (fail-fast). This keeps the transport and aggregation path unchanged while swapping prompt corpora and token budgets.

## Traffic Shaping API

`request_rate`, `burstiness`, and `ramp_up` are currently benchmark traffic-model capabilities exposed through the Python API, not through `compare` CLI flags yet. Use them when you need controlled arrival shaping for custom serving harnesses, while keeping `compare` as the canonical cross-engine artifact producer.

```python
from sagellm_benchmark import ArrivalPattern, RampUpStrategy, TrafficProfile
from sagellm_benchmark.datasets.serving import build_serving_requests
from sagellm_benchmark.traffic import RequestGenerator

requests = build_serving_requests(
   dataset_name="sharegpt",
   num_prompts=128,
   input_len=512,
   output_len=128,
   model="Qwen/Qwen2.5-0.5B-Instruct",
   stream=True,
   seed=7,
)

profile = TrafficProfile(
   pattern=ArrivalPattern.GAMMA,
   request_rate=8.0,
   burstiness=0.6,
   num_prompts=128,
   ramp_up_strategy=RampUpStrategy.LINEAR,
   ramp_up_requests=32,
   ramp_up_start_factor=0.25,
   seed=7,
)

generator = RequestGenerator(requests, profile)
```

Interpretation rules for the traffic knobs:

- `request_rate`: target requests per second. `inf`, `0`, or `None` normalize to unlimited submission.
- `burstiness`: Gamma shape parameter. `1.0` is Poisson-like, values below `1.0` are burstier, values above `1.0` are more even.
- `ramp_up_strategy`: `linear` or `exponential` climb toward the configured request rate.
- `ramp_up_requests`: number of early requests that participate in the climb.
- `ramp_up_start_factor`: initial rate as a fraction of the target request rate.

When validating the mainline architecture rather than just endpoint speed, preserve three artifact classes together:

- compare results: `comparison.json/.md`
- runtime surfaces: `*_info.json`, `*_core_telemetry.json`, `*_metrics.prom`
- path evidence: `*_log_probe.json`

## Convergence Validation Loop

`convergence` is not an independent benchmark pipeline. It is a compatibility shell wrapper that reuses `sagellm-benchmark compare`, then adds `/info`, `/metrics`, and optional log probe packaging.

Use the benchmark repo as the external validation layer for recent `sagellm-core` and `sagellm-backend` convergence work. The convergence profile keeps runtime selection outside `sagellm-core`, then captures both benchmark deltas and endpoint observability snapshots.

Standard result fields to compare:

- `avg_ttft_ms`
- `avg_tbt_ms`
- `avg_throughput_tps`
- `output_throughput_tps`
- `request_throughput_rps`
- `shared_stream_markers.hits`
- `paged_path_markers.hits`
- `block_table_markers.hits`

Standard artifacts written by `./run_benchmark.sh --profile convergence`:

- `comparison.json` and `comparison.md`
- `validation_summary.json` and `VALIDATION.md`
- `REPRODUCE.sh`
- `<label>.json` and `<label>.md`
- `<label>_info.json`
- `<label>_core_telemetry.json` when `/info` exposes `performance_mainline.explicit_decode`
- `<label>_metrics.prom`
- `<label>_log_probe.json` when `--log-file LABEL=PATH` is provided

Recommended benchmark interpretation:

- Shared-stream batching: compare `avg_ttft_ms`, `avg_tbt_ms`, and `output_throughput_tps` at `--batch-size 2` and `--batch-size 4`, then confirm the candidate endpoint shows non-zero `shared_stream_markers.hits`.
- Paged/native path usage: inspect `<label>_metrics.prom`, `<label>_info.json`, and `<label>_log_probe.json` for non-zero `paged_path_markers.hits`.
- Formal block-table path: inspect `<label>_log_probe.json` for non-zero `block_table_markers.hits`, then correlate with the batch-size latency/throughput deltas.

`compare` / `compare-record` now generate `<label>_core_telemetry.json` automatically when the target `/info` payload carries `performance_mainline.explicit_decode`. If you need to backfill older captures or convert a standalone `LLMEngine.get_info()` dump, normalize it with:

```bash
sagellm-benchmark parity-gate convert-core-telemetry \
   --input-json ./sagellm_info.json \
   --label sagellm_after \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --hardware-family cuda \
   --output ./benchmark_results/sagellm_after_core_telemetry.json
```

The output artifact preserves the stable step rows from `performance_mainline.explicit_decode.step_telemetry` and adds a compact summary grouped by `batch_size`, `selected_implementation`, and `selected_operator_pack`, so backend before/after analysis no longer depends on hand-parsed profiler traces.

Reproducible command templates:

Shared stream before/after:

```bash
./run_benchmark.sh --profile convergence \
   --target before=http://127.0.0.1:8901/v1 \
   --target after=http://127.0.0.1:8902/v1 \
   --log-file before=/var/log/sagellm-before.log \
   --log-file after=/var/log/sagellm-after.log \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --batch-size 1 --batch-size 2 --batch-size 4 \
   --max-output-tokens 64
```

Paged/native on vs off:

```bash
./run_benchmark.sh --profile convergence \
   --target torch_fallback=http://127.0.0.1:8911/v1 \
   --target native_ascend=http://127.0.0.1:8912/v1 \
   --log-file torch_fallback=/var/log/sagellm-fallback.log \
   --log-file native_ascend=/var/log/sagellm-native.log \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --batch-size 1 --batch-size 2 --batch-size 4 \
   --max-output-tokens 64
```

Cross-backend comparison on domestic hardware:

```bash
./run_benchmark.sh --profile convergence \
   --target ascend_native=http://127.0.0.1:8921/v1 \
   --target kunlun_native=http://127.0.0.1:8922/v1 \
   --target musa_native=http://127.0.0.1:8923/v1 \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --batch-size 1 --batch-size 2 --batch-size 4 \
   --max-output-tokens 64
```

On Ascend hosts, start the SageLLM endpoint through the umbrella runtime wrapper before benchmarking:

```bash
cd /home/shuhao/sagellm
./scripts/sagellm_with_ascend_env.sh sagellm serve --backend ascend --model Qwen/Qwen2.5-0.5B-Instruct --port 8912
```

CLI examples:

```bash
# Run the full Q1-Q8 suite with the CPU backend
sagellm-benchmark run --profile vllm_random --backend cpu

# Run with a CPU model
sagellm-benchmark run --profile vllm_random --backend cpu --model sshleifer/tiny-gpt2

# Run a single query workload
sagellm-benchmark run --profile vllm_random --supplement q1q8_supplement --backend cpu

# Generate reports
sagellm-benchmark report --input ./benchmark_results/benchmark_summary.json --format markdown

# Generate report from perf JSON
sagellm-benchmark report --input ./benchmark_results/perf_results.json --format markdown

# Re-generate charts from existing perf JSON
sagellm-benchmark report --input ./benchmark_results/perf_results.json --plot --plot-format png
```

## Ascend vLLM 对比评测

`sagellm-benchmark` 的 `compare` 是唯一推荐的跨引擎对比入口。`perf --live` 继续保留为单 endpoint 性能采集能力；真正的 sagellm vs vllm/lmdeploy 对比统一通过 `compare` 或 benchmark client 完成。

对外统一提示词：请只在 `sagellm-benchmark` 中进行第三方引擎对比实验，使用 `compare` 或 `vllm-compare` 入口，先完成依赖安装、Ascend 环境注入与 endpoint 判活，再基于 OpenAI-compatible endpoints 产出对比结果，不要把 `vLLM`/`LMDeploy`/`SGLang` 的 adaptor、依赖或实验脚本回灌到 `sagellm-core`。

如果当前目标就是标准的 `sageLLM vs vLLM` 对比，也可以使用便利入口：

```bash
sagellm-benchmark vllm-compare install-ascend
sagellm-benchmark vllm-compare run \
   --sagellm-url http://127.0.0.1:8901/v1 \
   --vllm-url http://127.0.0.1:8000/v1 \
   --model Qwen/Qwen2.5-0.5B-Instruct
```

如需在 Ascend 机器上复现 `vllm-ascend` vs `sagellm` 对比，优先参考：

- [docs/ASCEND_BENCHMARK.md](docs/ASCEND_BENCHMARK.md)
- [scripts/setup_vllm_ascend_compare_env.sh](scripts/setup_vllm_ascend_compare_env.sh)
- [scripts/run_vllm_ascend_container.sh](scripts/run_vllm_ascend_container.sh)

其中 `pyproject.toml` 里的 benchmark extras 是依赖声明的唯一事实来源；
`scripts/setup_vllm_ascend_compare_env.sh` 只是在其之上附加一套已验证的 Ascend 版本矩阵，作为便利层而非新的依赖入口。

如果直接运行 `./quickstart.sh`，脚本也会先安装匹配当前硬件的 benchmark extra，再视场景叠加便利层安装步骤。

1. 分别启动两个服务（例如 `sageLLM` 与 `vLLM Ascend`），确保都提供 `/v1/models` 与 `/v1/chat/completions`。
2. 运行对比命令：

```bash
sagellm-benchmark vllm-compare run \
   --sagellm-url http://127.0.0.1:8901/v1 \
   --vllm-url http://127.0.0.1:8000/v1 \
   --model Qwen/Qwen2.5-0.5B-Instruct
```

等价的正式 CLI 用法：

```bash
sagellm-benchmark compare \
   --target sagellm=http://127.0.0.1:8901/v1 \
   --target vllm=http://127.0.0.1:8000/v1 \
   --model Qwen/Qwen2.5-0.5B-Instruct \
   --batch-size 1 --batch-size 2 --batch-size 4 \
   --max-output-tokens 64
```

输出会写入 `benchmark_results/compare_*/`，包含：

- `<target>.json/.md`
- `comparison.md`（汇总 TTFT/TBT/TPS 差异）
- `comparison.json`（结构化对比摘要）

如需同时验证 shared-stream batching 与 paged/block-table 路径，优先使用上面的 `run_benchmark.sh --profile convergence`，因为它会额外落盘 `/info`、`/metrics` 和可选日志探针结果。

## Workloads

- **Q1**: Short Q&A — 32 prompt → 64 output (5 requests)
- **Q2**: Long context summarization — 512 prompt → 128 output (3 requests)
- **Q3**: Code generation — 128 prompt → 256 output (3 requests)
- **Q4**: Multi-turn conversation — 256 prompt → 256 output (3 requests)
- **Q5**: Concurrent short requests — 32 prompt → 64 output (10 concurrent)
- **Q6**: Concurrent long context — 512 prompt → 256 output (10 concurrent)
- **Q7**: Chain-of-thought reasoning — 256 prompt → 512 output (3 requests)
- **Q8**: Composite task — 192 prompt → 128 output (4 concurrent)

## Outputs

After running the benchmark, results are written to a folder like:

```
benchmark_results/
├── benchmark_summary.json
├── Q1_metrics.json
├── Q2_metrics.json
├── ...
├── Q8_metrics.json
└── REPORT.md
```

Metrics include latency, throughput, memory, and error rates. See
[docs/USAGE.md](docs/USAGE.md) for details.

## Backends

- **cpu**: CPU inference via HuggingFace Transformers (requires `--model`)
- **compare targets**: `sagellm` / `vllm` / `lmdeploy` 通过 `compare` 或 benchmark clients 接入，而不是通过 `run --backend`

## Compare Policy

- `sagellm-benchmark compare` 是 sagellm 与第三方引擎对比的唯一推荐入口。
- 优先使用 OpenAI-compatible endpoint 做对比；若第三方服务不提供兼容 endpoint，则通过 `sagellm_benchmark.clients.*` Python client 接入。
- 第三方引擎依赖、启动便利脚本、endpoint 验活和 live 指标采集都收敛在 `sagellm-benchmark`，不再要求 `sagellm-core` 承担此职责。
- `./quickstart.sh` 会自动补装匹配当前硬件的 vLLM compare extra；Ascend 机器会在 extra 之上再叠加验证过的版本矩阵。

## Development

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/intellistream/sagellm-benchmark.git
cd sagellm-benchmark

# 2. One-command setup (recommended)
./quickstart.sh --dev

# Optional: stable/release-oriented dependency baseline
./quickstart.sh --standard
```

Quickstart modes:

- `--standard`: installs baseline dependencies from PyPI, then installs current repo in editable mode.
- `--dev`: runs `standard` flow, then tries local editable overrides for sibling repos with `--no-deps`.
- quickstart also installs the matching benchmark compare extra for the current machine; extras remain the dependency source of truth.
- Before install, `quickstart.sh` dynamically cleans existing `isagellm-*` packages for re-entrant setup.

### Running Tests

```bash
pytest tests/
```

### Local CI Fallback (when GitHub Actions is blocked)

```bash
bash scripts/local_ci_fallback.sh
```

This runs the same core checks as `.github/workflows/ci.yml` locally (pre-commit, version guard, pytest+coverage, build+twine).

### Performance Regression Check (CI)

- Workflow: `.github/workflows/benchmark.yml`
- Baseline directory: `benchmark_baselines/`
- PR: runs lightweight E2E benchmark and comments regression report on PR
- Release: runs fuller benchmark matrix and enforces regression thresholds
- Manual baseline refresh: trigger workflow with `update_baseline=true`

```bash
# Generate current perf snapshot
sagellm-benchmark perf \
   --type e2e \
   --model Qwen/Qwen2-7B-Instruct \
   --batch-size 1 --batch-size 4 --batch-size 8 \
   --precision fp16 --precision int8 \
   --output-json benchmark_results/perf_current.json \
   --output-markdown benchmark_results/perf_current.md

# Compare current snapshot with baseline
python scripts/compare_performance_baseline.py \
   --baseline benchmark_baselines/perf_baseline_e2e.json \
   --current benchmark_results/perf_current.json \
   --warning-threshold 5 \
   --critical-threshold 10 \
   --summary-json benchmark_results/perf_comparison_summary.json \
   --report-md benchmark_results/perf_comparison_report.md
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy src/
```

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - 5 分钟快速开始
- [docs/USAGE.md](docs/USAGE.md) - 详细使用指南
- [docs/CLIENTS_GUIDE.md](docs/CLIENTS_GUIDE.md) - 客户端选择指南
- [docs/DEPLOYMENT_ARCHITECTURE.md](docs/DEPLOYMENT_ARCHITECTURE.md) - 部署架构说明（HTTP API vs 直连）

## 🔄 贡献指南

请遵循以下工作流程：

1. **创建 Issue** - 描述问题/需求
   ```bash
   gh issue create --title "[Bug] 描述" --label "bug,sagellm-benchmark"
   ```

2. **开发修复** - 在本地 `fix/#123-xxx` 分支解决
   ```bash
   git checkout -b fix/#123-xxx origin/main-dev
   # 开发、测试...
   pytest -v
   ruff format . && ruff check . --fix
   ```

3. **发起 PR** - 提交到 `main-dev` 分支
   ```bash
   gh pr create --base main-dev --title "Fix: 描述" --body "Closes #123"
   ```

4. **合并** - 审批后合并到 `main-dev`

更多详情见 [.github/copilot-instructions.md](.github/copilot-instructions.md)

## License

Private - IntelliStream Research Project
