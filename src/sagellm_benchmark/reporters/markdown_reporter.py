"""Markdown 报告生成器 - 输出 Markdown 格式的聚合指标与 Contract 验证结果。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sagellm_benchmark.types import AggregatedMetrics, ContractResult


class MarkdownReporter:
    """Markdown 格式报告生成器。

    生成包含表格、结论和详细检查结果的 Markdown 报告。
    """

    @staticmethod
    def _fmt_required(value: float | int, unit: str = "") -> str:
        if isinstance(value, float):
            return f"{value:.2f}{unit}"
        return f"{value}{unit}"

    @staticmethod
    def _fmt_optional(value: float | int, unit: str = "") -> str:
        if isinstance(value, float):
            if value <= 0:
                return "N/A"
            return f"{value:.2f}{unit}"
        if isinstance(value, int):
            if value <= 0:
                return "N/A"
            return f"{value}{unit}"
        return "N/A"

    @staticmethod
    def generate(
        metrics: AggregatedMetrics,
        contract: ContractResult | None = None,
        output_path: Path | str | None = None,
        title: str = "Benchmark Report",
        version: str = "unknown",
    ) -> str:
        """生成 Markdown 报告。

        Args:
            metrics: 聚合指标。
            contract: Contract 验证结果（可选）。
            output_path: 输出文件路径（None 则不保存）。
            title: 报告标题。
            version: Benchmark 版本。

        Returns:
            Markdown 字符串。
        """
        lines: list[str] = []

        # === 标题 ===
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Version**: {version}")
        lines.append("")

        # === 总览 ===
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Requests**: {metrics.total_requests}")
        lines.append(f"- **Successful**: {metrics.successful_requests}")
        lines.append(f"- **Failed**: {metrics.failed_requests}")
        lines.append(f"- **Error Rate**: {metrics.error_rate * 100:.2f}%")
        lines.append(f"- **Total Time**: {metrics.total_time_s:.2f}s")
        lines.append("")

        # === Contract 验证结果 ===
        if contract:
            lines.append("## Contract Validation")
            lines.append("")
            lines.append(f"**Result**: {contract.summary}")
            lines.append("")

            if contract.checks:
                lines.append("| Check | Status | Details |")
                lines.append("|-------|--------|---------|")

                for check_name, passed in contract.checks.items():
                    status = "✅ PASS" if passed else "❌ FAIL"
                    detail = contract.details.get(check_name, "N/A")
                    lines.append(f"| {check_name} | {status} | {detail} |")

                lines.append("")

        # === 延迟指标 ===
        lines.append("## Latency Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Avg TTFT | {MarkdownReporter._fmt_required(metrics.avg_ttft_ms, ' ms')} |")
        lines.append(f"| P50 TTFT | {MarkdownReporter._fmt_required(metrics.p50_ttft_ms, ' ms')} |")
        lines.append(f"| P95 TTFT | {MarkdownReporter._fmt_required(metrics.p95_ttft_ms, ' ms')} |")
        lines.append(f"| P99 TTFT | {MarkdownReporter._fmt_required(metrics.p99_ttft_ms, ' ms')} |")
        lines.append(f"| Avg TBT | {MarkdownReporter._fmt_required(metrics.avg_tbt_ms, ' ms')} |")
        lines.append(f"| Avg TPOT | {MarkdownReporter._fmt_optional(metrics.avg_tpot_ms, ' ms')} |")
        lines.append(f"| P50 TPOT | {MarkdownReporter._fmt_optional(metrics.p50_tpot_ms, ' ms')} |")
        lines.append(f"| P95 TPOT | {MarkdownReporter._fmt_optional(metrics.p95_tpot_ms, ' ms')} |")
        lines.append(f"| P99 TPOT | {MarkdownReporter._fmt_optional(metrics.p99_tpot_ms, ' ms')} |")
        lines.append(f"| Avg ITL | {MarkdownReporter._fmt_optional(metrics.avg_itl_ms, ' ms')} |")
        lines.append(f"| P50 ITL | {MarkdownReporter._fmt_optional(metrics.p50_itl_ms, ' ms')} |")
        lines.append(f"| P95 ITL | {MarkdownReporter._fmt_optional(metrics.p95_itl_ms, ' ms')} |")
        lines.append(f"| P99 ITL | {MarkdownReporter._fmt_optional(metrics.p99_itl_ms, ' ms')} |")
        lines.append(f"| Avg E2EL | {MarkdownReporter._fmt_optional(metrics.avg_e2el_ms, ' ms')} |")
        lines.append(f"| P50 E2EL | {MarkdownReporter._fmt_optional(metrics.p50_e2el_ms, ' ms')} |")
        lines.append(f"| P95 E2EL | {MarkdownReporter._fmt_optional(metrics.p95_e2el_ms, ' ms')} |")
        lines.append(f"| P99 E2EL | {MarkdownReporter._fmt_optional(metrics.p99_e2el_ms, ' ms')} |")
        lines.append("")

        # === 吞吐 ===
        lines.append("## Throughput")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(
            f"| Request Throughput | {MarkdownReporter._fmt_optional(metrics.request_throughput_rps, ' req/s')} |"
        )
        lines.append(
            f"| Input Throughput | {MarkdownReporter._fmt_optional(metrics.input_throughput_tps, ' tokens/s')} |"
        )
        lines.append(
            f"| Output Throughput | {MarkdownReporter._fmt_optional(metrics.output_throughput_tps, ' tokens/s')} |"
        )
        lines.append(
            f"| Total Throughput | {MarkdownReporter._fmt_required(metrics.total_throughput_tps, ' tokens/s')} |"
        )
        lines.append(
            f"| Avg Throughput | {MarkdownReporter._fmt_required(metrics.avg_throughput_tps, ' tokens/s')} |"
        )
        lines.append(
            f"| Total Input Tokens | {MarkdownReporter._fmt_optional(metrics.total_input_tokens)} |"
        )
        lines.append(
            f"| Total Output Tokens | {MarkdownReporter._fmt_optional(metrics.total_output_tokens)} |"
        )
        lines.append("")

        # === 内存 ===
        lines.append("## Memory")
        lines.append("")
        lines.append(f"- **Peak Memory**: {metrics.peak_mem_mb} MB")
        lines.append("")

        # === KV Cache ===
        lines.append("## KV Cache")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total KV Used Tokens | {metrics.total_kv_used_tokens} |")
        lines.append(f"| Total KV Used Bytes | {metrics.total_kv_used_bytes} |")
        lines.append(f"| Avg Prefix Hit Rate | {metrics.avg_prefix_hit_rate * 100:.2f}% |")
        lines.append(f"| Total Evict Count | {metrics.total_evict_count} |")
        lines.append(f"| Total Evict Time | {metrics.total_evict_ms:.2f} ms |")
        lines.append("")

        # === Speculative ===
        if metrics.avg_spec_accept_rate > 0:
            lines.append("## Speculative Decoding")
            lines.append("")
            lines.append(
                f"- **Avg Speculative Accept Rate**: {metrics.avg_spec_accept_rate * 100:.2f}%"
            )
            lines.append("")

        # === 脚注 ===
        lines.append("---")
        lines.append("")
        lines.append("*Generated by sagellm-benchmark*")
        lines.append("")

        # 合并为字符串
        markdown_str = "\n".join(lines)

        # 保存到文件
        if output_path:
            Path(output_path).write_text(markdown_str, encoding="utf-8")

        return markdown_str
