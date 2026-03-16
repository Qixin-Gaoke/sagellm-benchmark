# sagellm-benchmark Design

> Last updated: 2026-03-14
> Status: active

## 1. Positioning

`sagellm-benchmark` owns benchmark orchestration, artifact generation, publish/export boundaries, and third-party engine comparison flows. It does not own engine execution logic, protocol definitions, or serving runtime internals.

Benchmark outputs are now **canonical-first**:

- Benchmark execution writes `*.canonical.json` first.
- Legacy leaderboard files are derived later as a compatibility export boundary.
- Publish/HF/website sync consume that explicit compatibility boundary instead of acting as an internal source of truth.

## 2. Entrypoint Responsibilities

| Entrypoint | Role | Status | Notes |
|---|---|---|---|
| `sagellm-benchmark run` | Canonical local workload benchmark pipeline | Mainline | Produces local workload `execution_result` canonical artifacts. |
| `sagellm-benchmark compare` | Canonical live multi-endpoint benchmark pipeline | Mainline | Produces one canonical `execution_result` per target plus one canonical comparison summary. |
| `sagellm-benchmark vllm-compare install-ascend` | Validated Ascend compare environment setup | Setup helper | Does not define a benchmark execution path. |
| `./run_benchmark.sh --profile quick` | Shell convenience wrapper for local default run | Compatibility layer | Reuses `sagellm-benchmark run`; retained for scripting/backward compatibility. |
| `./run_benchmark.sh --profile convergence` | Shell convenience wrapper for compare plus extra probes | Compatibility layer | Reuses `sagellm-benchmark compare` and adds `/info`/`/metrics`/log probes plus validation packaging. |
| `sagellm-benchmark compare-record` | Single-target capture helper for constrained environments | Compatibility layer | Reuses the canonical compare target pipeline; useful when endpoints cannot coexist. |
| `sagellm-benchmark compare-offline` | Offline summary helper over captured results | Compatibility layer | Does not define a new benchmark pipeline. |

## 3. Recommended Paths

Only two benchmark execution paths are recommended:

1. `sagellm-benchmark run` for local workload benchmarks.
2. `sagellm-benchmark compare` for live endpoint comparison.

Use `sagellm-benchmark vllm-compare install-ascend` only for validated Ascend environment bring-up. Use `run_benchmark.sh` only when an existing shell workflow depends on it.

## 4. Canonical-First Pipeline

### 4.1 Local run

`run` writes:

- `config.json`
- `benchmark_summary.json`
- `*_metrics.json`
- `*.canonical.json` as the benchmark source of truth
- compatibility exports: `*_leaderboard.json`, `leaderboard_manifest.json`

### 4.2 Live compare

`compare` writes:

- `<label>.json` and `<label>.md` for raw capture/reporting
- `<label>.canonical.json` as the canonical per-target result
- `<label>.parity.json` and runtime evidence artifacts
- `comparison.json`, `comparison.md`, `comparison.canonical.json`
- compatibility exports: `*_leaderboard.json`, `leaderboard_manifest.json`

### 4.3 Publish boundary

Publish is explicit:

- `run --publish`
- `compare --publish`

Publish first regenerates the compatibility export boundary from canonical artifacts, then uploads canonical per-entry files plus website-facing HF snapshots.

## 5. Compatibility Boundary

The following files are **compatibility artifacts**, not the internal benchmark source of truth:

- `*_leaderboard.json`
- `leaderboard_manifest.json`
- `leaderboard_single.json`
- `leaderboard_multi.json`
- `last_updated.json`

Rules:

- Benchmark execution logic must not depend on these files.
- Export/upload/sync flows may consume them only at the compatibility boundary.
- If a compatibility artifact is missing or invalid, export must fail fast rather than silently guessing fields.

## 6. Chat-First and No Fork Rules

Benchmark protocol behavior stays aligned with SageLLM serving boundaries:

- benchmark compare paths target OpenAI-compatible chat flows first
- do not introduce a completions-only benchmark fork
- `/v1/completions` remains compatibility-only and must not become a separate benchmark mainline
- no silent fallback from canonical compare to a second legacy leaderboard-first path
- no silent fallback from requested datasets or runtime evidence requirements

## 7. Why Old Leaderboard-First Wiring Was Removed

The old shape allowed `run` and `compare` to both write leaderboard files directly during execution. That caused two structural problems:

1. benchmark semantics were split across canonical artifacts and leaderboard export rows
2. shell wrappers and publish flows could accidentally treat compatibility files as the primary result model

The current design removes that branch:

- execution paths write canonical artifacts only
- compatibility exports are generated afterward by one shared exporter
- warnings make it explicit when a user invokes a compatibility or convenience layer

## 8. Operational Guidance

### Recommended commands

```bash
# Local canonical workload benchmark
sagellm-benchmark run --workload all --backend cpu

# Canonical live endpoint compare
sagellm-benchmark compare \
  --target sagellm=http://127.0.0.1:8901/v1 \
  --target vllm=http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --hardware-family cuda

# Validated Ascend compare environment setup
sagellm-benchmark vllm-compare install-ascend
```

### Compatibility wrappers

```bash
# Backward-compatible shell wrapper over `run`
./run_benchmark.sh --profile quick

# Backward-compatible shell wrapper over `compare` plus probes
./run_benchmark.sh --profile convergence \
  --target sagellm=http://127.0.0.1:8901/v1 \
  --target vllm=http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct
```# sagellm-benchmark 设计文档

> **最后更新**：2026-01-17  
> **版本**：v0.1.0.2  
> **状态**：开发中

## 1. 背景与定位
`isagellm-benchmark` 是 **顶层 Benchmark Runner**，用于统一执行各类性能测试与验收，不承担引擎核心逻辑。

- **负责**：数据集/Workload 定义、测试编排、指标聚合、报告输出、Demo Contract 校验。
- **不负责**：引擎实现、协议定义（由 `isagellm-protocol` 提供）、业务推理逻辑。

## 2. 设计目标
1. **Protocol-First**：指标与请求/响应严格对齐 `sagellm_protocol.Metrics` 与 `Request/Response`。
2. **CPU-First**：CI 无 GPU 也可完整跑通。
3. **可独立扩展**：数据集、Runner、报告三条链路可独立开发。
4. **Demo Contract 验证**：支持 Year1/2/3 合规性检查。

## 3. 当前实现状态

### 已实现模块
| 文件 | 状态 | 说明 |
|------|------|------|
| `workloads.py` | ✅ 已实现 | `WorkloadType`, `WorkloadConfig`, `YEAR1_WORKLOADS` |
| `runner.py` | ✅ 已实现 | `BenchmarkRunner`, `BenchmarkConfig` |
| `metrics.py` | ✅ 已实现 | `BenchmarkMetrics`, `MetricsCollector` |
| `cli.py` | ✅ 已实现 | CLI 命令 `run`, `report` |

### 待完善模块
| 模块 | 状态 | 任务书 |
|------|------|--------|
| `datasets/` | 🔲 待开发 | TASK_A_DATASETS.md |
| `clients/` | 🔲 待开发 | TASK_B_RUNNER.md |
| `reporters/` | 🔲 待开发 | TASK_C_METRICS_REPORT.md |
| `contract.py` | 🔲 待开发 | TASK_C_METRICS_REPORT.md |

## 4. 系统分层与职责

```
┌──────────────────────────────────────────────────────────┐
│                  sagellm-benchmark (顶层)                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ Datasets     │   │ Runner       │   │ Metrics+Report│ │
│  │ (模块A)      │   │ (模块B)      │   │ (模块C)       │ │
│  │ 🔲待开发     │   │ ✅已有基础    │   │ ✅已有基础    │ │
│  └──────────────┘   └──────────────┘   └──────────────┘  │
│        ▲                   ▲                   ▲          │
│   WorkloadConfig     BenchmarkMetrics    Report/Contract  │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│    sagellm-backend (engine: cpu/hf-cuda)                 │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                 isagellm-protocol (底层)                 │
│  - Metrics, Request, Response, Timestamps 等类型定义    │
└──────────────────────────────────────────────────────────┘
```

## 5. 现有核心数据结构

### 5.1 WorkloadConfig (workloads.py - ✅已实现)
```python
class WorkloadType(str, Enum):
    SHORT = "short"
    LONG = "long"
    STRESS = "stress"

@dataclass
class WorkloadConfig:
    name: str
    workload_type: WorkloadType
    prompt: str
    prompt_tokens: int
    max_tokens: int
    num_requests: int = 1
    concurrent: bool = False
    temperature: float | None = None
    top_p: float = 1.0
    extra_params: dict[str, Any] = field(default_factory=dict)
```

### 5.2 BenchmarkMetrics (metrics.py - ✅已实现)
```python
@dataclass
class BenchmarkMetrics:
    # 延迟指标
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    avg_tbt_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    avg_throughput_tps: float = 0.0

    # 内存
    peak_mem_mb: int = 0
    avg_mem_mb: float = 0.0

    # 错误率
    error_rate: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # KV Cache
    kv_used_tokens: int = 0
    kv_used_bytes: int = 0
    prefix_hit_rate: float = 0.0
    evict_count: int = 0
    evict_ms: float = 0.0

    # Speculative
    spec_accept_rate: float = 0.0

    # 时间
    total_time_s: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
```

### 5.3 BenchmarkRunner (runner.py - ✅已实现)
```python
@dataclass
class BenchmarkConfig:
    engine: Any  # BaseEngine instance from sagellm-backend
    workloads: list[WorkloadConfig]
    output_dir: Path = Path("./benchmark_results")
    verbose: bool = False

class BenchmarkRunner:
    async def run(self) -> dict[str, BenchmarkMetrics]: ...
    async def _run_workload(self, workload: WorkloadConfig) -> BenchmarkMetrics: ...
```

## 6. CLI 使用 (✅已实现)

```bash
# 运行 M1 所有 workload (CPU 后端)
sagellm-benchmark run --workload m1 --backend cpu --model sshleifer/tiny-gpt2

# 运行单个 workload
sagellm-benchmark run --workload short --backend cpu

# 查看报告
sagellm-benchmark report --input ./benchmark_results/benchmark_summary.json --format table
sagellm-benchmark report --input ./benchmark_results/benchmark_summary.json --format markdown
```

## 7. 待开发模块拆分（三人并行）

### 模块A：Datasets 扩展（开发者A）
**现状**：`workloads.py` 已有 `YEAR1_WORKLOADS` 预定义  
**待开发**：
- `datasets/` 目录：支持 ShareGPT/Random/自定义数据集
- 数据集采样接口
- Year2/Year3 workloads

**详见**：`docs/TASK_A_DATASETS.md`

---

### 模块B：Clients 扩展（开发者B）
**现状**：`runner.py` 已集成 `sagellm-backend` engine  
**待开发**：
- `clients/` 目录：抽象 Client 层
- OpenAI 兼容 Client（对接 sagellm-gateway）
- 外部后端 Client（vLLM/LMDeploy 对比测试）

**详见**：`docs/TASK_B_RUNNER.md`

---

### 模块C：Reporters & Contract（开发者C）
**现状**：`metrics.py` 已有 `BenchmarkMetrics`，CLI 有 table/json/markdown 输出  
**待开发**：
- `reporters/` 目录：独立报告生成器
- `contract.py`：Year1/2/3 Demo Contract 验证器
- HTML 报告（可选）

**详见**：`docs/TASK_C_METRICS_REPORT.md`

## 8. 依赖关系

```
isagellm-benchmark (本仓库)
    │
    ├── isagellm (umbrella，可选)
    │       └── isagellm-backend (engine: cpu/hf-cuda)
    │       └── isagellm-protocol (Metrics, Request, Response)
    │
    └── 直接依赖
            └── isagellm-backend (必须)
            └── isagellm-protocol (必须)
```

## 9. 里程碑计划
- **M1** ✅：基础框架（workloads/runner/metrics/cli）
- **M2**：三模块扩展（datasets/clients/reporters）
- **M3**：Year1 Demo Contract 验证跑通
- **M4**：Year2/3 Contract + 外部后端对比

## 10. 风险与约束
- `Metrics` 必须严格对齐 `sagellm_protocol.Metrics` 字段
- CPU 模式必须可在 CI 运行（无 GPU）
- 不引入 engine 依赖到 datasets/reporters 模块

## 11. Canonical Artifact Direction

- 统一 benchmark artifact schema 设计见 [CANONICAL_RESULT_SCHEMA.md](CANONICAL_RESULT_SCHEMA.md)
- 后续 `run` / `compare` / `convergence` 应先落 canonical artifact，再导出 leaderboard JSON
- leaderboard 不应继续作为 benchmark 内部唯一事实源

## 12. Publish Boundary

- Local benchmark outputs are responsible for producing:
    - `*.canonical.json`
    - per-entry `*_leaderboard.json`
    - `leaderboard_manifest.json`
- `upload-hf` must consume only the standard export boundary above and publish:
    - canonical per-entry files under dataset `canonical/...`
    - website-facing HF snapshots `leaderboard_single.json`, `leaderboard_multi.json`, `last_updated.json`
- `sagellm-website` should treat the HF snapshots as the primary runtime data source.
- Local website sync is only an offline compatibility path and must consume `leaderboard_manifest.json`, not compare raw directories or ad hoc file globs.
