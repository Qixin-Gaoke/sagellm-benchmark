"""Tests for RankingDashboard (#8)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sagellm_benchmark.dashboard import RankingDashboard
from sagellm_benchmark.dashboard.ranking import LeaderboardEntry


@pytest.fixture
def benchmark_results_dir():
    """Create a temp dir with sample benchmark result files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)

        # Write a perf_results.json file (row format)
        data = {
            "kind": "e2e",
            "rows": [
                {
                    "model": "Qwen2-7B",
                    "scenario": "vllm_random_b1",
                    "backend": "sagellm-cpu",
                    "hardware": "Intel Xeon",
                    "ttft_ms": 25.0,
                    "tbt_ms": 5.0,
                    "throughput_tps": 80.0,
                    "latency_p50_ms": 30.0,
                    "latency_p99_ms": 60.0,
                    "memory_mb": 512.0,
                },
                {
                    "model": "Qwen2-7B",
                    "scenario": "vllm_sharegpt_b1",
                    "backend": "sagellm-cpu",
                    "hardware": "Intel Xeon",
                    "ttft_ms": 80.0,
                    "tbt_ms": 10.0,
                    "throughput_tps": 40.0,
                    "latency_p50_ms": 120.0,
                    "latency_p99_ms": 200.0,
                    "memory_mb": 512.0,
                },
            ],
        }
        (p / "perf_results.json").write_text(json.dumps(data), encoding="utf-8")

        # Write a second file (aggregated metrics format)
        data2 = {
            "model": "tiny-gpt2",
            "workload": "vllm_random_b1",
            "backend": "vllm",
            "metrics": {
                "avg_ttft_ms": 12.0,
                "avg_tbt_ms": 3.0,
                "output_throughput_tps": 120.0,
                "p50_ttft_ms": 11.0,
                "p99_ttft_ms": 20.0,
                "peak_mem_mb": 256,
            },
        }
        (p / "vllm_results.json").write_text(json.dumps(data2), encoding="utf-8")

        yield p


def test_dashboard_load(benchmark_results_dir: Path) -> None:
    db = RankingDashboard(results_dir=benchmark_results_dir)
    db.load()
    assert len(db._entries) == 3  # 2 rows + 1 aggregated


def test_dashboard_entries_have_model(benchmark_results_dir: Path) -> None:
    db = RankingDashboard(results_dir=benchmark_results_dir)
    db.load()
    models = {e.model for e in db._entries}
    assert "Qwen2-7B" in models


def test_dashboard_generate_returns_html(benchmark_results_dir: Path) -> None:
    db = RankingDashboard(results_dir=benchmark_results_dir)
    html = db.generate()
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "Leaderboard" in html


def test_dashboard_generate_contains_scenarios(benchmark_results_dir: Path) -> None:
    db = RankingDashboard(results_dir=benchmark_results_dir)
    html = db.generate()
    assert "vllm_random_b1" in html
    assert "vllm_sharegpt_b1" in html


def test_dashboard_saves_file(benchmark_results_dir: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        tmp = Path(f.name)
    try:
        db = RankingDashboard(results_dir=benchmark_results_dir)
        db.generate(output_path=tmp)
        assert tmp.exists()
        content = tmp.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
    finally:
        tmp.unlink()


def test_dashboard_empty_dir() -> None:
    """Empty directory should not crash — just produce empty leaderboard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = RankingDashboard(results_dir=tmpdir)
        db.load()
        assert db._entries == []


def test_dashboard_skips_invalid_json() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "bad.json").write_text("not valid json", encoding="utf-8")
        db = RankingDashboard(results_dir=p)
        db.load()  # should not raise
        assert db._entries == []


def test_dashboard_leaderboard_entry() -> None:
    e = LeaderboardEntry(
        model="test-model",
        scenario="test-scenario",
        throughput_tps=100.0,
    )
    assert e.model == "test-model"
    assert e.backend == "unknown"
    assert e.throughput_tps == 100.0
