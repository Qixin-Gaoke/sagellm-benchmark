"""测试 Reporters（JSON/Markdown/Table）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sagellm_benchmark.reporters import JSONReporter, MarkdownReporter, TableReporter
from sagellm_benchmark.types import AggregatedMetrics, ContractResult, ContractVersion


@pytest.fixture
def sample_aggregated_metrics() -> AggregatedMetrics:
    """创建示例 AggregatedMetrics。"""
    return AggregatedMetrics(
        avg_ttft_ms=20.0,
        p50_ttft_ms=20.0,
        p95_ttft_ms=30.0,
        p99_ttft_ms=30.0,
        avg_tbt_ms=4.0,
        avg_tpot_ms=3.5,
        p50_tpot_ms=3.2,
        p95_tpot_ms=4.1,
        p99_tpot_ms=4.6,
        avg_itl_ms=2.8,
        p50_itl_ms=2.5,
        p95_itl_ms=3.6,
        p99_itl_ms=4.0,
        avg_e2el_ms=42.0,
        p50_e2el_ms=40.0,
        p95_e2el_ms=55.0,
        p99_e2el_ms=60.0,
        avg_throughput_tps=80.0,
        total_throughput_tps=100.0,
        request_throughput_rps=5.0,
        input_throughput_tps=20.0,
        output_throughput_tps=80.0,
        total_input_tokens=200,
        total_output_tokens=800,
        total_requests=5,
        successful_requests=5,
        failed_requests=0,
        error_rate=0.0,
        peak_mem_mb=2048,
        total_kv_used_tokens=960,
        total_kv_used_bytes=15360,
        avg_prefix_hit_rate=0.84,
        total_evict_count=10,
        total_evict_ms=5.0,
        avg_spec_accept_rate=0.72,
        total_time_s=10.0,
        start_time=1000.0,
        end_time=1010.0,
    )


@pytest.fixture
def sample_contract_result() -> ContractResult:
    """创建示例 ContractResult。"""
    return ContractResult(
        passed=True,
        version=ContractVersion.YEAR1,
        checks={
            "ttft_ms": True,
            "tbt_ms": True,
            "throughput_tps": True,
            "error_rate": True,
        },
        details={
            "ttft_ms": "TTFT: 20.00ms ≤ 100ms",
            "tbt_ms": "TBT: 4.00ms ≤ 20ms",
            "throughput_tps": "Throughput: 80.00 tokens/s ≥ 50 tokens/s",
            "error_rate": "Error Rate: 0.00% ≤ 5.00%",
        },
        summary="Contract YEAR1: ✅ PASSED (4/4 checks passed)",
    )


def test_json_reporter_basic(
    sample_aggregated_metrics: AggregatedMetrics,
    tmp_path: Path,
) -> None:
    """测试 JSON 报告生成。"""
    output_file = tmp_path / "report.json"

    json_str = JSONReporter.generate(
        metrics=sample_aggregated_metrics,
        output_path=output_file,
        version="0.1.0.2",
    )

    # 验证返回的 JSON 字符串
    assert isinstance(json_str, str)
    data = json.loads(json_str)

    assert "metrics" in data
    assert data["metrics"]["avg_ttft_ms"] == 20.0
    assert data["version"] == "0.1.0.2"

    # 验证文件写入
    assert output_file.exists()
    loaded_data = json.loads(output_file.read_text())
    assert loaded_data == data


def test_json_reporter_with_contract(
    sample_aggregated_metrics: AggregatedMetrics,
    sample_contract_result: ContractResult,
    tmp_path: Path,
) -> None:
    """测试包含 Contract 的 JSON 报告。"""
    output_file = tmp_path / "report_with_contract.json"

    json_str = JSONReporter.generate(
        metrics=sample_aggregated_metrics,
        contract=sample_contract_result,
        output_path=output_file,
    )

    data = json.loads(json_str)

    assert "metrics" in data
    assert "contract" in data
    assert data["contract"]["passed"] is True
    assert data["contract"]["version"] == "year1"


def test_json_reporter_load(
    sample_aggregated_metrics: AggregatedMetrics,
    tmp_path: Path,
) -> None:
    """测试从文件加载 JSON 报告。"""
    output_file = tmp_path / "report.json"

    JSONReporter.generate(
        metrics=sample_aggregated_metrics,
        output_path=output_file,
    )

    loaded = JSONReporter.load(output_file)

    assert loaded["metrics"]["avg_ttft_ms"] == 20.0
    assert loaded["metrics"]["total_requests"] == 5


def test_markdown_reporter_basic(
    sample_aggregated_metrics: AggregatedMetrics,
    tmp_path: Path,
) -> None:
    """测试 Markdown 报告生成。"""
    output_file = tmp_path / "report.md"

    markdown_str = MarkdownReporter.generate(
        metrics=sample_aggregated_metrics,
        output_path=output_file,
        title="Test Benchmark Report",
        version="0.1.0.2",
    )

    # 验证返回的 Markdown 字符串
    assert isinstance(markdown_str, str)
    assert "# Test Benchmark Report" in markdown_str
    assert "**Version**: 0.1.0.2" in markdown_str
    assert "## Summary" in markdown_str
    assert "Total Requests**: 5" in markdown_str
    assert "## Latency Metrics" in markdown_str
    assert "Avg TTFT | 20.00 ms" in markdown_str
    assert "Avg ITL | 2.80 ms" in markdown_str
    assert "Avg E2EL | 42.00 ms" in markdown_str
    assert "Request Throughput | 5.00 req/s" in markdown_str
    assert "Total Output Tokens | 800" in markdown_str

    # 验证文件写入
    assert output_file.exists()
    content = output_file.read_text()
    assert content == markdown_str


def test_markdown_reporter_with_contract(
    sample_aggregated_metrics: AggregatedMetrics,
    sample_contract_result: ContractResult,
    tmp_path: Path,
) -> None:
    """测试包含 Contract 的 Markdown 报告。"""
    output_file = tmp_path / "report_with_contract.md"

    markdown_str = MarkdownReporter.generate(
        metrics=sample_aggregated_metrics,
        contract=sample_contract_result,
        output_path=output_file,
    )

    assert "## Contract Validation" in markdown_str
    assert "Contract YEAR1: ✅ PASSED" in markdown_str
    assert "| ttft_ms | ✅ PASS |" in markdown_str


def test_markdown_reporter_honest_degrades_missing_stream_metrics(tmp_path: Path) -> None:
    output_file = tmp_path / "report_missing_stream.md"
    metrics = AggregatedMetrics(
        avg_ttft_ms=20.0,
        p50_ttft_ms=20.0,
        p95_ttft_ms=30.0,
        p99_ttft_ms=30.0,
        avg_tbt_ms=4.0,
        avg_throughput_tps=80.0,
        total_throughput_tps=100.0,
        total_requests=5,
        successful_requests=5,
        peak_mem_mb=2048,
    )

    markdown_str = MarkdownReporter.generate(metrics=metrics, output_path=output_file)

    assert "Avg ITL | N/A" in markdown_str
    assert "Avg E2EL | N/A" in markdown_str
    assert "Request Throughput | N/A" in markdown_str
    assert output_file.exists()


def test_table_reporter_plain_text(
    sample_aggregated_metrics: AggregatedMetrics,
    sample_contract_result: ContractResult,
    capsys,
) -> None:
    """测试 Table 报告生成（plain text fallback）。"""
    # 强制使用 plain text（模拟 Rich 未安装）
    TableReporter._generate_plain_text(
        metrics=sample_aggregated_metrics,
        contract=sample_contract_result,
        show_contract=True,
    )

    captured = capsys.readouterr()

    assert "Benchmark Summary" in captured.out
    assert "Total Requests: 5" in captured.out
    assert "Contract Validation (YEAR1)" in captured.out
    assert "ttft_ms: PASS" in captured.out
    assert "Latency Metrics" in captured.out
    assert "Avg TTFT: 20.00 ms" in captured.out
    assert "Avg ITL: 2.80 ms" in captured.out
    assert "Avg E2EL: 42.00 ms" in captured.out


def test_table_reporter_with_rich(
    sample_aggregated_metrics: AggregatedMetrics,
    sample_contract_result: ContractResult,
) -> None:
    """测试 Table 报告生成（Rich）。"""
    try:
        import rich  # noqa: F401
    except ImportError:
        pytest.skip("Rich not installed")

    # 不验证输出内容（Rich 输出包含 ANSI 转义序列），只验证不抛异常
    TableReporter.generate(
        metrics=sample_aggregated_metrics,
        contract=sample_contract_result,
        show_contract=True,
    )
