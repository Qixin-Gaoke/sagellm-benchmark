"""Tests for HTMLReporter (#2, #12)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sagellm_benchmark.reporters import HTMLReporter
from sagellm_benchmark.types import AggregatedMetrics, ContractResult, ContractVersion


@pytest.fixture
def sample_metrics() -> AggregatedMetrics:
    return AggregatedMetrics(
        avg_ttft_ms=25.0,
        p50_ttft_ms=22.0,
        p95_ttft_ms=40.0,
        p99_ttft_ms=55.0,
        avg_tbt_ms=5.0,
        avg_tpot_ms=4.5,
        avg_itl_ms=3.0,
        p95_itl_ms=4.5,
        avg_e2el_ms=48.0,
        p95_e2el_ms=60.0,
        avg_throughput_tps=75.0,
        total_throughput_tps=90.0,
        input_throughput_tps=30.0,
        output_throughput_tps=75.0,
        request_throughput_rps=3.5,
        total_input_tokens=150,
        total_output_tokens=375,
        total_requests=10,
        successful_requests=10,
        failed_requests=0,
        error_rate=0.0,
        peak_mem_mb=1024,
        total_kv_used_tokens=1280,
        total_kv_used_bytes=20480,
        avg_prefix_hit_rate=0.65,
        total_evict_count=3,
        total_evict_ms=2.0,
        avg_spec_accept_rate=0.0,
        total_time_s=5.0,
        start_time=1000.0,
        end_time=1005.0,
    )


@pytest.fixture
def sample_contract() -> ContractResult:
    return ContractResult(
        version=ContractVersion.YEAR1,
        passed=True,
        summary="All checks passed",
        checks={"ttft": True, "throughput": True},
        details={"ttft": "22ms < 100ms", "throughput": "3.5 req/s > 1.0"},
    )


def test_html_reporter_returns_string(sample_metrics: AggregatedMetrics) -> None:
    html = HTMLReporter.generate(sample_metrics)
    assert isinstance(html, str)
    assert len(html) > 500


def test_html_reporter_contains_doctype(sample_metrics: AggregatedMetrics) -> None:
    html = HTMLReporter.generate(sample_metrics)
    assert "<!DOCTYPE html>" in html


def test_html_reporter_contains_chartjs(sample_metrics: AggregatedMetrics) -> None:
    html = HTMLReporter.generate(sample_metrics)
    assert "chart.js" in html.lower() or "Chart.js" in html


def test_html_reporter_contains_metrics(sample_metrics: AggregatedMetrics) -> None:
    html = HTMLReporter.generate(sample_metrics)
    # Key metric values should appear in the output
    assert "25.0" in html or "25" in html  # avg_ttft_ms
    assert "10" in html  # total_requests
    assert "Avg E2EL" in html
    assert "Avg ITL" in html
    assert "Total Output Tokens" in html


def test_html_reporter_with_contract(
    sample_metrics: AggregatedMetrics, sample_contract: ContractResult
) -> None:
    html = HTMLReporter.generate(sample_metrics, contract=sample_contract)
    assert "Contract Validation" in html
    assert "All checks passed" in html


def test_html_reporter_with_title(sample_metrics: AggregatedMetrics) -> None:
    html = HTMLReporter.generate(sample_metrics, title="My Custom Report")
    assert "My Custom Report" in html


def test_html_reporter_saves_file(sample_metrics: AggregatedMetrics) -> None:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        tmp = Path(f.name)
    try:
        HTMLReporter.generate(sample_metrics, output_path=tmp)
        assert tmp.exists()
        content = tmp.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
    finally:
        tmp.unlink()


def test_html_reporter_generate_multi(sample_metrics: AggregatedMetrics) -> None:
    """Multi-run comparison report."""
    metrics2 = AggregatedMetrics(
        avg_ttft_ms=35.0,
        p50_ttft_ms=32.0,
        p95_ttft_ms=50.0,
        p99_ttft_ms=65.0,
        avg_tbt_ms=6.0,
        avg_tpot_ms=5.5,
        avg_throughput_tps=65.0,
        total_throughput_tps=80.0,
        input_throughput_tps=25.0,
        output_throughput_tps=65.0,
        request_throughput_rps=2.8,
        total_requests=8,
        successful_requests=8,
        failed_requests=0,
        error_rate=0.0,
        peak_mem_mb=1500,
        total_kv_used_tokens=1000,
        total_kv_used_bytes=16000,
        avg_prefix_hit_rate=0.5,
        total_evict_count=5,
        total_evict_ms=3.0,
        avg_spec_accept_rate=0.0,
        total_time_s=6.5,
        start_time=2000.0,
        end_time=2006.5,
    )
    html = HTMLReporter.generate_multi(
        runs=[sample_metrics, metrics2],
        labels=["SageLLM-CPU", "SageLLM-v2"],
    )
    assert "SageLLM-CPU" in html
    assert "SageLLM-v2" in html
    assert "Comparison" in html


def test_html_reporter_multi_empty_raises(sample_metrics: AggregatedMetrics) -> None:
    with pytest.raises(ValueError, match="At least one run"):
        HTMLReporter.generate_multi(runs=[])


def test_html_reporter_exported_from_reporters_module() -> None:
    from sagellm_benchmark.reporters import HTMLReporter as HTMLReporterAliased  # noqa: F401

    assert HTMLReporterAliased is HTMLReporter


def test_html_reporter_honest_degrades_missing_stream_metrics() -> None:
    metrics = AggregatedMetrics(
        avg_ttft_ms=25.0,
        p50_ttft_ms=22.0,
        p95_ttft_ms=40.0,
        p99_ttft_ms=55.0,
        avg_tbt_ms=5.0,
        avg_throughput_tps=75.0,
        total_throughput_tps=90.0,
        total_requests=10,
        successful_requests=10,
        peak_mem_mb=1024,
    )

    html = HTMLReporter.generate(metrics)

    assert "N/A" in html
    assert "Avg E2EL" in html
    assert "Output Throughput" in html
