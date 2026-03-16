"""公共数据类型定义 - 符合 INTERFACE_CONTRACT.md 契约。

此模块定义了三个模块间交互所需的所有公共类型：
- BenchmarkRequest: A→B 传递
- BenchmarkResult: B→C 传递
- WorkloadSpec: A 内部使用
- AggregatedMetrics: C 输出
- ContractResult: C 输出
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sagellm_protocol import Metrics


class WorkloadType(StrEnum):
    """Workload 类型枚举。"""

    SHORT = "short"  # 短输入短输出
    LONG = "long"  # 长输入中输出
    STRESS = "stress"  # 并发压力测试


class ContractVersion(StrEnum):
    """Demo Contract 版本。"""

    YEAR1 = "year1"
    YEAR2 = "year2"
    YEAR3 = "year3"


@dataclass
class BenchmarkRequest:
    """Benchmark 请求，用于 Dataset→Runner 传递。

    Attributes:
        prompt: 输入 prompt 文本。
        max_tokens: 最大生成 token 数。
        request_id: 请求唯一标识（UUID）。
        model: 模型名称，Protocol 必填字段。
        stream: 是否流式输出，Protocol 必填字段。
        temperature: 采样温度，None 表示使用模型默认。
        top_p: Nucleus 采样参数。
        kv_budget_tokens: KV 缓存预算（tokens）。
    """

    prompt: str
    max_tokens: int
    request_id: str
    # Protocol 必填字段，提供默认值
    model: str = "default"
    stream: bool = False
    # 可选采样参数
    temperature: float | None = None
    top_p: float | None = None
    # KV 缓存配置
    kv_budget_tokens: int | None = None


@dataclass
class WorkloadSpec:
    """Workload 规格描述，用于 Dataset 内部生成请求。

    Attributes:
        name: Workload 名称标识。
        workload_type: Workload 类型（short/long/stress）。
        prompt_len: 期望的 prompt 长度（tokens/字符，允许 ±10% 误差）。
        output_len: 期望的输出长度（tokens）。
        num_requests: 请求数量。
        concurrent: 是否并发执行。
        kv_budget_tokens: KV 缓存预算。
    """

    name: str
    workload_type: WorkloadType
    prompt_len: int
    output_len: int
    num_requests: int
    concurrent: bool = False
    kv_budget_tokens: int | None = None


@dataclass
class BenchmarkResult:
    """单请求执行结果，用于 Runner→Aggregator 传递。

    Attributes:
        request_id: 请求唯一标识。
        success: 是否执行成功。
        error: 错误信息（失败时）。
        metrics: Protocol 定义的 Metrics 对象。
        output_text: 生成的文本输出。
        output_tokens: 输出 token 数。
        prompt_tokens: 输入 token 数。
        is_warmup: 是否为 warmup 请求结果；聚合时默认排除。
        batch_total_time_s: batch 模式下的统一墙钟总时长（秒）。
    """

    request_id: str
    success: bool
    error: str | None
    # 直接使用 Protocol Metrics，运行时导入避免循环依赖
    metrics: Metrics | None = None
    # 可选输出信息
    output_text: str = ""
    output_tokens: int = 0
    prompt_tokens: int = 0
    # 新增：benchmark 层面的延迟记录
    itl_list: list[float] = field(default_factory=list)  # 逐 token 延迟（ms）
    e2e_latency_ms: float = 0.0  # 端到端延迟（从发送到完成）
    is_warmup: bool = False
    batch_total_time_s: float = 0.0


@dataclass
class AggregatedMetrics:
    """聚合指标（多请求统计），模块 C 输出。

    Attributes:
        avg_ttft_ms: 平均首 token 延迟（ms）。
        p50_ttft_ms: P50 首 token 延迟。
        p95_ttft_ms: P95 首 token 延迟。
        p99_ttft_ms: P99 首 token 延迟。
        avg_tbt_ms: 平均 token 间延迟。
        avg_tpot_ms: 平均每输出 token 时间。
        avg_throughput_tps: 平均吞吐（tokens/s）。
        total_throughput_tps: 总吞吐。
        total_requests: 总请求数。
        successful_requests: 成功请求数。
        failed_requests: 失败请求数。
        error_rate: 错误率。
        peak_mem_mb: 峰值内存（MB）。
        total_kv_used_tokens: KV 缓存使用总 tokens。
        total_kv_used_bytes: KV 缓存使用总字节。
        avg_prefix_hit_rate: 平均前缀命中率。
        total_evict_count: 总驱逐次数。
        total_evict_ms: 总驱逐耗时（ms）。
        avg_spec_accept_rate: 平均推测解码接受率。
        total_time_s: 总耗时（s）。
        start_time: 开始时间戳。
        end_time: 结束时间戳。
    """

    # TTFT 延迟指标
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0  # 新增：TTFT 标准差

    # TBT 延迟指标
    avg_tbt_ms: float = 0.0

    # TPOT 延迟指标
    avg_tpot_ms: float = 0.0
    p50_tpot_ms: float = 0.0  # 新增：TPOT percentiles
    p95_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0

    # 新增：ITL (Inter-Token Latency) percentiles
    avg_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    std_itl_ms: float = 0.0

    # 新增：E2E Latency percentiles
    avg_e2el_ms: float = 0.0
    p50_e2el_ms: float = 0.0
    p95_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0
    std_e2el_ms: float = 0.0

    # 吞吐
    avg_throughput_tps: float = 0.0
    total_throughput_tps: float = 0.0

    # 新增：对标 vLLM/SGLang 的吞吐量指标
    request_throughput_rps: float = 0.0  # 请求吞吐量 (requests/s)
    input_throughput_tps: float = 0.0  # 输入 token 吞吐量 (tokens/s)
    output_throughput_tps: float = 0.0  # 输出 token 吞吐量 (tokens/s)
    # total_throughput_tps 已有，表示 (input + output) / total_time_s

    # Token 统计（用于计算吞吐量）
    total_input_tokens: int = 0  # 总输入 tokens
    total_output_tokens: int = 0  # 总输出 tokens

    # 错误率
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0

    # 内存
    peak_mem_mb: int = 0

    # KV Cache
    total_kv_used_tokens: int = 0
    total_kv_used_bytes: int = 0
    avg_prefix_hit_rate: float = 0.0
    total_evict_count: int = 0
    total_evict_ms: float = 0.0

    # Speculative
    avg_spec_accept_rate: float = 0.0

    # 时间
    total_time_s: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ContractResult:
    """Contract 验证结果，模块 C 输出。

    Attributes:
        passed: 是否通过验证。
        version: Contract 版本。
        checks: 每项检查的结果。
        details: 每项检查的详细说明。
        summary: 总结文本。
    """

    passed: bool
    version: ContractVersion
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    summary: str = ""
