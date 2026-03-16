"""HTML 报告生成器 - 输出交互式 HTML 格式的聚合指标与 Contract 验证结果。

生成包含内嵌 Chart.js 可视化图表的单文件 HTML 报告，无需外部依赖。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sagellm_benchmark.types import AggregatedMetrics, ContractResult


# ---------------------------------------------------------------------------
# Chart.js CDN version pinned for reproducibility
# ---------------------------------------------------------------------------
_CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"


class HTMLReporter:
    """交互式 HTML 格式报告生成器。

    生成单文件 HTML 报告，包含：
    - TTFT/TBT/TPOT P50/P95/P99 百分位数对比图 (Chart.js 柱状图)
    - 吞吐量汇总图 (Chart.js 条形图)
    - KV Cache 指标图
    - 所有指标的汇总表格
    - Contract 验证结果
    - 多次运行对比 (传入 runs 列表时启用)

    Example::

        from sagellm_benchmark.reporters import HTMLReporter

        html = HTMLReporter.generate(metrics, contract, output_path="report.html")
    """

    @staticmethod
    def _optional_metric_value(value: float | int | None) -> float | int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)) and value <= 0:
            return None
        return value

    @staticmethod
    def _format_metric(value: float | int | None, unit: str = "") -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}{unit}"
        return f"{value}{unit}"

    @staticmethod
    def generate(
        metrics: AggregatedMetrics,
        contract: ContractResult | None = None,
        output_path: Path | str | None = None,
        title: str = "SageLLM Benchmark Report",
        version: str = "unknown",
        extra_info: dict[str, Any] | None = None,
    ) -> str:
        """生成单次运行的 HTML 报告。

        Args:
            metrics: 聚合指标。
            contract: Contract 验证结果（可选）。
            output_path: 输出文件路径（None 则不保存）。
            title: 报告标题。
            version: Benchmark 版本。
            extra_info: 额外信息字典（显示在页面顶部）。

        Returns:
            HTML 字符串。
        """
        return HTMLReporter.generate_multi(
            runs=[metrics],
            labels=["Run 1"],
            contracts=[contract],
            output_path=output_path,
            title=title,
            version=version,
            extra_info=extra_info,
        )

    @staticmethod
    def generate_multi(
        runs: list[AggregatedMetrics],
        labels: list[str] | None = None,
        contracts: list[ContractResult | None] | None = None,
        output_path: Path | str | None = None,
        title: str = "SageLLM Benchmark Comparison Report",
        version: str = "unknown",
        extra_info: dict[str, Any] | None = None,
    ) -> str:
        """生成多次运行对比的 HTML 报告。

        Args:
            runs: 多次运行的聚合指标列表。
            labels: 每次运行的标签（默认 Run 1, Run 2, ...）。
            contracts: 每次运行的 Contract 验证结果（可选）。
            output_path: 输出文件路径（None 则不保存）。
            title: 报告标题。
            version: Benchmark 版本。
            extra_info: 额外信息字典。

        Returns:
            HTML 字符串。
        """
        if not runs:
            raise ValueError("At least one run is required")

        effective_labels = labels or [f"Run {i + 1}" for i in range(len(runs))]
        effective_contracts = contracts or [None] * len(runs)

        html = HTMLReporter._build_html(
            runs=runs,
            labels=effective_labels,
            contracts=effective_contracts,
            title=title,
            version=version,
            extra_info=extra_info or {},
        )

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")

        return html

    @staticmethod
    def _build_html(  # noqa: C901
        runs: list[AggregatedMetrics],
        labels: list[str],
        contracts: list[ContractResult | None],
        title: str,
        version: str,
        extra_info: dict[str, Any],
    ) -> str:
        """Build the complete HTML string."""

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        multi = len(runs) > 1

        # -------------------------------------------------------------------
        # Build chart data
        # -------------------------------------------------------------------
        # Latency chart: TTFT P50/P95/P99 per run
        latency_data = {
            "labels": labels,
            "datasets": [
                {
                    "label": "TTFT P50 (ms)",
                    "data": [r.p50_ttft_ms for r in runs],
                    "backgroundColor": "rgba(54, 162, 235, 0.7)",
                },
                {
                    "label": "TTFT P95 (ms)",
                    "data": [r.p95_ttft_ms for r in runs],
                    "backgroundColor": "rgba(255, 159, 64, 0.7)",
                },
                {
                    "label": "TTFT P99 (ms)",
                    "data": [r.p99_ttft_ms for r in runs],
                    "backgroundColor": "rgba(255, 99, 132, 0.7)",
                },
                {
                    "label": "Avg TBT (ms)",
                    "data": [r.avg_tbt_ms for r in runs],
                    "backgroundColor": "rgba(75, 192, 192, 0.7)",
                },
                {
                    "label": "Avg TPOT (ms)",
                    "data": [HTMLReporter._optional_metric_value(r.avg_tpot_ms) for r in runs],
                    "backgroundColor": "rgba(99, 102, 241, 0.7)",
                },
                {
                    "label": "Avg ITL (ms)",
                    "data": [HTMLReporter._optional_metric_value(r.avg_itl_ms) for r in runs],
                    "backgroundColor": "rgba(16, 185, 129, 0.7)",
                },
                {
                    "label": "Avg E2EL (ms)",
                    "data": [HTMLReporter._optional_metric_value(r.avg_e2el_ms) for r in runs],
                    "backgroundColor": "rgba(245, 158, 11, 0.7)",
                },
            ],
        }

        # Throughput chart
        throughput_data = {
            "labels": labels,
            "datasets": [
                {
                    "label": "Request Throughput (req/s)",
                    "data": [r.request_throughput_rps for r in runs],
                    "backgroundColor": "rgba(153, 102, 255, 0.7)",
                },
                {
                    "label": "Output Throughput (tokens/s)",
                    "data": [r.output_throughput_tps for r in runs],
                    "backgroundColor": "rgba(255, 205, 86, 0.7)",
                },
            ],
        }

        # KV Cache chart
        kv_data = {
            "labels": labels,
            "datasets": [
                {
                    "label": "Prefix Hit Rate (%)",
                    "data": [r.avg_prefix_hit_rate * 100 for r in runs],
                    "backgroundColor": "rgba(54, 162, 235, 0.7)",
                },
                {
                    "label": "Evict Count",
                    "data": [r.total_evict_count for r in runs],
                    "backgroundColor": "rgba(255, 99, 132, 0.7)",
                },
            ],
        }

        # -------------------------------------------------------------------
        # Build metrics table rows
        # -------------------------------------------------------------------

        def _fmt(v: float | int | None, unit: str = "") -> str:
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return f"{v:.2f}{unit}"
            return f"{v}{unit}"

        comparison_header = "".join(f"<th>{lbl}</th>" for lbl in labels)

        def _metric_row(label: str, values: list[str]) -> str:
            cells = "".join(f"<td>{v}</td>" for v in values)
            return f"<tr><td><b>{label}</b></td>{cells}</tr>"

        table_rows = "\n".join(
            [
                _metric_row("Total Requests", [str(r.total_requests) for r in runs]),
                _metric_row("Successful", [str(r.successful_requests) for r in runs]),
                _metric_row("Failed", [str(r.failed_requests) for r in runs]),
                _metric_row("Error Rate", [_fmt(r.error_rate * 100, "%") for r in runs]),
                _metric_row("Total Time", [_fmt(r.total_time_s, "s") for r in runs]),
                _metric_row("Avg TTFT", [_fmt(r.avg_ttft_ms, " ms") for r in runs]),
                _metric_row("P50 TTFT", [_fmt(r.p50_ttft_ms, " ms") for r in runs]),
                _metric_row("P95 TTFT", [_fmt(r.p95_ttft_ms, " ms") for r in runs]),
                _metric_row("P99 TTFT", [_fmt(r.p99_ttft_ms, " ms") for r in runs]),
                _metric_row("Avg TBT", [_fmt(r.avg_tbt_ms, " ms") for r in runs]),
                _metric_row(
                    "Avg TPOT",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.avg_tpot_ms),
                            " ms",
                        )
                        for r in runs
                    ],
                ),
                _metric_row(
                    "Avg ITL",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.avg_itl_ms),
                            " ms",
                        )
                        for r in runs
                    ],
                ),
                _metric_row(
                    "P95 ITL",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.p95_itl_ms),
                            " ms",
                        )
                        for r in runs
                    ],
                ),
                _metric_row(
                    "Avg E2EL",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.avg_e2el_ms),
                            " ms",
                        )
                        for r in runs
                    ],
                ),
                _metric_row(
                    "P95 E2EL",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.p95_e2el_ms),
                            " ms",
                        )
                        for r in runs
                    ],
                ),
                _metric_row(
                    "Request Throughput",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.request_throughput_rps),
                            " req/s",
                        )
                        for r in runs
                    ],
                ),
                _metric_row(
                    "Input Throughput",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.input_throughput_tps),
                            " tokens/s",
                        )
                        for r in runs
                    ],
                ),
                _metric_row(
                    "Output Throughput",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.output_throughput_tps),
                            " tokens/s",
                        )
                        for r in runs
                    ],
                ),
                _metric_row(
                    "Total Throughput",
                    [_fmt(r.total_throughput_tps, " tokens/s") for r in runs],
                ),
                _metric_row(
                    "Total Input Tokens",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.total_input_tokens)
                        )
                        for r in runs
                    ],
                ),
                _metric_row(
                    "Total Output Tokens",
                    [
                        HTMLReporter._format_metric(
                            HTMLReporter._optional_metric_value(r.total_output_tokens)
                        )
                        for r in runs
                    ],
                ),
                _metric_row("Peak Memory", [_fmt(r.peak_mem_mb, " MB") for r in runs]),
                _metric_row(
                    "Prefix Hit Rate",
                    [_fmt(r.avg_prefix_hit_rate * 100, "%") for r in runs],
                ),
                _metric_row(
                    "Total KV Used Tokens",
                    [str(r.total_kv_used_tokens) for r in runs],
                ),
                _metric_row("Evict Count", [str(r.total_evict_count) for r in runs]),
                _metric_row(
                    "Spec Accept Rate",
                    [_fmt(r.avg_spec_accept_rate * 100, "%") for r in runs],
                ),
            ]
        )

        # -------------------------------------------------------------------
        # Contract section
        # -------------------------------------------------------------------
        contract_sections = ""
        for i, (lbl, contract) in enumerate(zip(labels, contracts)):
            if contract is None:
                continue
            rows = ""
            for check_name, passed in (contract.checks or {}).items():
                status = "✅ PASS" if passed else "❌ FAIL"
                detail = (contract.details or {}).get(check_name, "N/A")
                rows += f"<tr><td>{check_name}</td><td>{status}</td><td>{detail}</td></tr>\n"
            contract_sections += f"""
            <h3>Contract Validation — {lbl}</h3>
            <p><b>Result:</b> {contract.summary}</p>
            {"<table><tr><th>Check</th><th>Status</th><th>Details</th></tr>" + rows + "</table>" if rows else ""}
            """

        # -------------------------------------------------------------------
        # Extra info
        # -------------------------------------------------------------------
        extra_html = ""
        if extra_info:
            items = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in extra_info.items())
            extra_html = f"<ul>{items}</ul>"

        multi_note = "(Multi-Run Comparison)" if multi else ""

        # -------------------------------------------------------------------
        # Assemble HTML
        # -------------------------------------------------------------------
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <script src="{_CHARTJS_CDN}"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
            background: #f5f7fa;
            color: #333;
            margin: 0;
            padding: 0;
        }}
        header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #fff;
            padding: 2rem 2.5rem;
        }}
        header h1 {{ margin: 0 0 0.5rem; font-size: 1.8rem; }}
        header p {{ margin: 0; opacity: 0.8; font-size: 0.95rem; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }}
        .card {{
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .card h2 {{ margin-top: 0; font-size: 1.2rem; color: #0f3460; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 1.5rem;
        }}
        .chart-box {{ background: #fff; border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 1.5rem; }}
        .chart-box h3 {{ margin-top: 0; font-size: 1rem; color: #555; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
        th {{ background: #0f3460; color: #fff; padding: 0.6rem 1rem; text-align: left; }}
        td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #e8e8e8; }}
        tr:hover td {{ background: #f0f4ff; }}
        .badge-pass {{ color: #2e7d32; font-weight: 600; }}
        .badge-fail {{ color: #c62828; font-weight: 600; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
        .info-item {{ background: #f0f4ff; border-radius: 8px; padding: 0.8rem 1rem; }}
        .info-item .label {{ font-size: 0.78rem; color: #666; margin-bottom: 0.2rem; }}
        .info-item .value {{ font-size: 1.1rem; font-weight: 600; color: #0f3460; }}
        footer {{ text-align: center; color: #999; font-size: 0.8rem; padding: 2rem; }}
    </style>
</head>
<body>
<header>
    <h1>📊 {title} {multi_note}</h1>
    <p>Generated: {generated_at} &nbsp;|&nbsp; Version: {version}</p>
    {extra_html}
</header>
<div class="container">

    <!-- Key Stats -->
    <div class="card">
        <h2>Key Metrics (Latest Run)</h2>
        <div class="info-grid">
            <div class="info-item">
                <div class="label">Total Requests</div>
                <div class="value">{runs[-1].total_requests}</div>
            </div>
            <div class="info-item">
                <div class="label">Avg TTFT</div>
                <div class="value">{runs[-1].avg_ttft_ms:.1f} ms</div>
            </div>
            <div class="info-item">
                <div class="label">P99 TTFT</div>
                <div class="value">{runs[-1].p99_ttft_ms:.1f} ms</div>
            </div>
            <div class="info-item">
                <div class="label">Output Throughput</div>
                <div class="value">{HTMLReporter._format_metric(HTMLReporter._optional_metric_value(runs[-1].output_throughput_tps), " tok/s")}</div>
            </div>
            <div class="info-item">
                <div class="label">Request Throughput</div>
                <div class="value">{HTMLReporter._format_metric(HTMLReporter._optional_metric_value(runs[-1].request_throughput_rps), " req/s")}</div>
            </div>
            <div class="info-item">
                <div class="label">Avg E2EL</div>
                <div class="value">{HTMLReporter._format_metric(HTMLReporter._optional_metric_value(runs[-1].avg_e2el_ms), " ms")}</div>
            </div>
            <div class="info-item">
                <div class="label">Error Rate</div>
                <div class="value">{runs[-1].error_rate * 100:.2f}%</div>
            </div>
        </div>
    </div>

    <!-- Charts -->
    <div class="charts-grid">
        <div class="chart-box">
            <h3>Latency Distribution (ms)</h3>
            <canvas id="latencyChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>Throughput</h3>
            <canvas id="throughputChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>KV Cache</h3>
            <canvas id="kvChart"></canvas>
        </div>
    </div>

    <!-- Metrics Table -->
    <div class="card">
        <h2>Full Metrics Table</h2>
        <table>
            <thead><tr><th>Metric</th>{comparison_header}</tr></thead>
            <tbody>
            {table_rows}
            </tbody>
        </table>
    </div>

    <!-- Contract -->
    {f'<div class="card"><h2>Contract Validation</h2>{contract_sections}</div>' if contract_sections else ""}

</div>
<footer>Generated by sagellm-benchmark &nbsp;|&nbsp; {generated_at}</footer>

<script>
const latencyData = {json.dumps(latency_data)};
const throughputData = {json.dumps(throughput_data)};
const kvData = {json.dumps(kv_data)};

const chartDefaults = {{
    type: 'bar',
    options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{ y: {{ beginAtZero: true }} }}
    }}
}};

new Chart(document.getElementById('latencyChart'), {{
    ...chartDefaults, data: latencyData
}});
new Chart(document.getElementById('throughputChart'), {{
    ...chartDefaults, data: throughputData
}});
new Chart(document.getElementById('kvChart'), {{
    ...chartDefaults, data: kvData
}});
</script>
</body>
</html>
"""
        return html
