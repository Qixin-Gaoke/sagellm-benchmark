from __future__ import annotations

import json

from sagellm_benchmark.baseline import BaselineManager, BenchmarkBaselineConfig
from sagellm_benchmark.regression import RegressionDetector, extract_metrics, render_markdown


def test_baseline_manager_update(tmp_path):
    baseline_path = tmp_path / "perf_baseline.json"
    manager = BaselineManager(baseline_path=baseline_path)

    payload = {
        "kind": "e2e",
        "summary": {
            "avg_ttft_ms": 50.0,
            "avg_tbt_ms": 10.0,
            "avg_throughput_tps": 100.0,
        },
    }

    manager.update(payload)
    saved = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert saved["summary"]["avg_ttft_ms"] == 50.0
    assert "baseline_updated_at" in saved["metadata"]
    assert saved["metadata"]["benchmark_contract"]["enabled"] is False


def test_baseline_config_parses_contract_switches():
    config = BenchmarkBaselineConfig.from_env(
        {"SAGELLM_BENCH_CR": "1", "SAGELLM_BENCH_CR_KILL": "0"}
    )

    assert config.enabled is True
    assert config.kill_switch_active is False


def test_regression_detector_expected_change():
    baseline = {
        "summary": {
            "avg_ttft_ms": 50.0,
            "avg_tbt_ms": 10.0,
            "avg_throughput_tps": 100.0,
            "peak_mem_mb": 4000.0,
            "error_rate": 0.01,
        }
    }
    current = {
        "summary": {
            "avg_ttft_ms": 58.0,
            "avg_tbt_ms": 10.6,
            "avg_throughput_tps": 96.0,
            "peak_mem_mb": 4300.0,
            "error_rate": 0.03,
        }
    }

    detector = RegressionDetector(
        warning_threshold_pct=5.0,
        critical_threshold_pct=10.0,
        expected_changes={"avg_ttft_ms"},
    )
    summary = detector.compare(baseline, current)

    assert summary["metrics"]["avg_ttft_ms"]["status"] == "expected-change"
    assert summary["metrics"]["avg_tbt_ms"]["status"] == "warning"
    assert summary["metrics"]["peak_mem_mb"]["status"] == "warning"
    assert summary["metrics"]["error_rate"]["status"] == "critical"
    assert summary["metrics"]["peak_mem_mb"]["direction"] == "lower-is-better"
    assert summary["overall_status"] == "critical"

    report = render_markdown(summary)
    assert "allowlisted" in report


def test_extract_metrics_from_rows():
    payload = {
        "rows": [
            {"ttft_ms": 10, "tbt_ms": 5, "throughput_tps": 20},
            {"ttft_ms": 30, "tbt_ms": 7, "throughput_tps": 10, "memory_mb": 123, "ok": False},
        ]
    }

    metrics = extract_metrics(payload)
    assert metrics["avg_ttft_ms"] == 20.0
    assert metrics["avg_tbt_ms"] == 6.0
    assert metrics["avg_throughput_tps"] == 15.0
    assert metrics["peak_mem_mb"] == 123.0
    assert metrics["error_rate"] == 0.5
