"""Common test fixtures for sagellm-benchmark."""

from __future__ import annotations

import os

# Keep pytest CPU-safe on Ascend hosts before any helper imports torch.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest


@pytest.fixture
def sample_metrics():
    """Provide sample benchmark metrics."""
    return {
        "avg_ttft_ms": 45.2,
        "p50_ttft_ms": 40.0,
        "p95_ttft_ms": 55.0,
        "p99_ttft_ms": 60.0,
        "avg_tbt_ms": 12.5,
        "avg_tpot_ms": 12.5,
        "avg_throughput_tps": 80.0,
        "peak_mem_mb": 24576,
        "avg_mem_mb": 20480.0,
        "error_rate": 0.02,
        "total_requests": 10,
        "successful_requests": 9,
        "failed_requests": 1,
        "kv_used_tokens": 4096,
        "kv_used_bytes": 134217728,
        "prefix_hit_rate": 0.85,
        "evict_count": 3,
        "evict_ms": 2.1,
        "spec_accept_rate": 0.72,
    }


@pytest.fixture
def sample_summary():
    """Provide sample benchmark summary."""
    return {
        "workloads": {
            "short": {
                "avg_ttft_ms": 45.2,
                "avg_throughput_tps": 80.0,
                "error_rate": 0.0,
            },
            "long": {
                "avg_ttft_ms": 52.3,
                "avg_throughput_tps": 75.0,
                "error_rate": 0.0,
            },
            "stress": {
                "avg_ttft_ms": 65.8,
                "avg_throughput_tps": 60.0,
                "error_rate": 0.1,
            },
        },
        "overall": {
            "total_tests": 3,
            "passed_tests": 3,
            "failed_tests": 0,
        },
    }
