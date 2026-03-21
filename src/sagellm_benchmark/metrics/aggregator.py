"""指标聚合器 - 将多个 BenchmarkResult 聚合为 AggregatedMetrics。

符合 INTERFACE_CONTRACT.md §4 定义。
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sagellm_benchmark.types import AggregatedMetrics, BenchmarkResult


class MetricsAggregator:
    """指标聚合器，将多个请求的结果聚合为统计指标。

    聚合规则：
    - 延迟类（ttft_ms, tbt_ms, tpot_ms）：取平均 + 百分位
    - 内存（peak_mem_mb）：取 max
    - KV Cache 计数（evict_count, kv_used_tokens）：取 sum
    - 比率类（prefix_hit_rate, spec_accept_rate）：取平均
    """

    @staticmethod
    def aggregate(results: list[BenchmarkResult]) -> AggregatedMetrics:
        """聚合多个 BenchmarkResult 为 AggregatedMetrics。

        Args:
            results: BenchmarkResult 列表。

        Returns:
            聚合后的 AggregatedMetrics。

        Note:
            - 如果所有请求都失败，返回空 AggregatedMetrics
            - 百分位使用排序后的索引法
        """
        from sagellm_benchmark.types import AggregatedMetrics

        # 初始化空指标
        aggregated = AggregatedMetrics()

        if not results:
            return aggregated

        included_results = [r for r in results if not getattr(r, "is_warmup", False)]
        if not included_results:
            return aggregated

        # 统计总数
        aggregated.total_requests = len(included_results)
        successful = [r for r in included_results if r.success and r.metrics is not None]
        aggregated.successful_requests = len(successful)
        aggregated.failed_requests = aggregated.total_requests - aggregated.successful_requests
        aggregated.error_rate = (
            aggregated.failed_requests / aggregated.total_requests
            if aggregated.total_requests > 0
            else 0.0
        )

        # 如果全部失败，直接返回
        if not successful:
            return aggregated

        # 收集时间戳（从 timestamps 对象中获取），batch 模式优先使用统一墙钟时长。
        start_times = [
            r.metrics.timestamps.queued_at
            for r in successful
            if r.metrics.timestamps is not None and r.metrics.timestamps.queued_at > 0
        ]
        end_times = [
            r.metrics.timestamps.completed_at
            for r in successful
            if r.metrics.timestamps is not None and r.metrics.timestamps.completed_at > 0
        ]

        batch_total_times = [
            MetricsAggregator._batch_total_time_s(r)
            for r in successful
            if MetricsAggregator._batch_total_time_s(r) > 0
        ]

        if batch_total_times:
            aggregated.total_time_s = max(batch_total_times)
            if start_times:
                aggregated.start_time = min(start_times)
                aggregated.end_time = aggregated.start_time + aggregated.total_time_s
        elif start_times and end_times:
            aggregated.start_time = min(start_times)
            aggregated.end_time = max(end_times)
            aggregated.total_time_s = aggregated.end_time - aggregated.start_time

        # === 延迟指标 ===
        ttft_samples = [r.metrics.ttft_ms for r in successful if r.metrics.ttft_ms > 0]
        tbt_samples = [r.metrics.tbt_ms for r in successful if r.metrics.tbt_ms > 0]
        tpot_samples = [r.metrics.tpot_ms for r in successful if r.metrics.tpot_ms > 0]

        if ttft_samples:
            aggregated.avg_ttft_ms = statistics.mean(ttft_samples)
            aggregated.p50_ttft_ms = MetricsAggregator._percentile(ttft_samples, 0.50)
            aggregated.p95_ttft_ms = MetricsAggregator._percentile(ttft_samples, 0.95)
            aggregated.p99_ttft_ms = MetricsAggregator._percentile(ttft_samples, 0.99)
            if len(ttft_samples) > 1:
                aggregated.std_ttft_ms = statistics.stdev(ttft_samples)

        if tbt_samples:
            aggregated.avg_tbt_ms = statistics.mean(tbt_samples)

        if tpot_samples:
            aggregated.avg_tpot_ms = statistics.mean(tpot_samples)
            aggregated.p50_tpot_ms = MetricsAggregator._percentile(tpot_samples, 0.50)
            aggregated.p95_tpot_ms = MetricsAggregator._percentile(tpot_samples, 0.95)
            aggregated.p99_tpot_ms = MetricsAggregator._percentile(tpot_samples, 0.99)
            if len(tpot_samples) > 1:
                aggregated.std_tpot_ms = statistics.stdev(tpot_samples)

        # === ITL 指标（展平所有请求的 itl_list）===
        all_itl: list[float] = []
        for r in successful:
            itl_samples = r.itl_list or list(getattr(r.metrics, "itl_list", []) or [])
            all_itl.extend(sample for sample in itl_samples if sample > 0)

        if all_itl:
            aggregated.avg_itl_ms = statistics.mean(all_itl)
            aggregated.p50_itl_ms = MetricsAggregator._percentile(all_itl, 0.50)
            aggregated.p95_itl_ms = MetricsAggregator._percentile(all_itl, 0.95)
            aggregated.p99_itl_ms = MetricsAggregator._percentile(all_itl, 0.99)
            if len(all_itl) > 1:
                aggregated.std_itl_ms = statistics.stdev(all_itl)

        # === E2E Latency 指标 ===
        e2el_samples = [MetricsAggregator._resolve_e2e_latency_ms(r) for r in successful]
        e2el_samples = [sample for sample in e2el_samples if sample > 0]

        if e2el_samples:
            aggregated.avg_e2el_ms = statistics.mean(e2el_samples)
            aggregated.p50_e2el_ms = MetricsAggregator._percentile(e2el_samples, 0.50)
            aggregated.p95_e2el_ms = MetricsAggregator._percentile(e2el_samples, 0.95)
            aggregated.p99_e2el_ms = MetricsAggregator._percentile(e2el_samples, 0.99)
            if len(e2el_samples) > 1:
                aggregated.std_e2el_ms = statistics.stdev(e2el_samples)

        # === 吞吐 ===
        throughput_samples = [
            r.metrics.throughput_tps for r in successful if r.metrics.throughput_tps > 0
        ]

        if throughput_samples:
            aggregated.avg_throughput_tps = statistics.mean(throughput_samples)

        # 新增：Token 统计与对标吞吐量指标
        total_input_tokens = sum(r.prompt_tokens for r in successful)
        total_output_tokens = sum(r.output_tokens for r in successful)
        aggregated.total_input_tokens = total_input_tokens
        aggregated.total_output_tokens = total_output_tokens

        # 计算对标 vLLM/SGLang 的吞吐量指标
        if aggregated.total_time_s > 0:
            # Request throughput (requests/s)
            aggregated.request_throughput_rps = (
                aggregated.successful_requests / aggregated.total_time_s
            )
            # Input throughput (tokens/s)
            aggregated.input_throughput_tps = total_input_tokens / aggregated.total_time_s
            # Output throughput (tokens/s)
            aggregated.output_throughput_tps = total_output_tokens / aggregated.total_time_s
            # Total throughput = (input + output) / total_time_s
            aggregated.total_throughput_tps = (
                total_input_tokens + total_output_tokens
            ) / aggregated.total_time_s

        # === 内存（取 max）===
        mem_samples = [r.metrics.peak_mem_mb for r in successful if r.metrics.peak_mem_mb > 0]
        if mem_samples:
            aggregated.peak_mem_mb = max(mem_samples)

        # === KV Cache（取 sum/avg）===
        aggregated.total_kv_used_tokens = sum(r.metrics.kv_used_tokens for r in successful)
        aggregated.total_kv_used_bytes = sum(r.metrics.kv_used_bytes for r in successful)

        prefix_hit_samples = [
            r.metrics.prefix_hit_rate for r in successful if r.metrics.prefix_hit_rate >= 0
        ]
        if prefix_hit_samples:
            aggregated.avg_prefix_hit_rate = statistics.mean(prefix_hit_samples)

        aggregated.total_evict_count = sum(r.metrics.evict_count for r in successful)
        aggregated.total_evict_ms = sum(r.metrics.evict_ms for r in successful)

        # === Speculative（取 avg）===
        spec_samples = [
            r.metrics.spec_accept_rate for r in successful if r.metrics.spec_accept_rate >= 0
        ]
        if spec_samples:
            aggregated.avg_spec_accept_rate = statistics.mean(spec_samples)

        return aggregated

    @staticmethod
    def _batch_total_time_s(result: BenchmarkResult) -> float:
        batch_total_time_s = getattr(result, "batch_total_time_s", 0.0)
        if batch_total_time_s > 0:
            return float(batch_total_time_s)

        legacy_batch_total_time_s = getattr(result, "_batch_total_time_s", 0.0)
        if legacy_batch_total_time_s > 0:
            return float(legacy_batch_total_time_s)

        return 0.0

    @staticmethod
    def _resolve_e2e_latency_ms(result: BenchmarkResult) -> float:
        if result.e2e_latency_ms > 0:
            return result.e2e_latency_ms

        timestamps = result.metrics.timestamps if result.metrics is not None else None
        if timestamps is None:
            return 0.0

        if timestamps.queued_at > 0 and timestamps.completed_at > 0:
            return (timestamps.completed_at - timestamps.queued_at) * 1000.0

        return 0.0

    @staticmethod
    def _percentile(samples: list[float], p: float) -> float:
        """计算百分位。

        Args:
            samples: 样本列表。
            p: 百分位（0-1）。

        Returns:
            百分位值。
        """
        if not samples:
            return 0.0

        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        index = int(n * p)

        # 边界保护
        if index >= n:
            index = n - 1

        return sorted_samples[index]
