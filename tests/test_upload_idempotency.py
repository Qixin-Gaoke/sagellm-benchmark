"""Tests for idempotent leaderboard upload helpers."""

from __future__ import annotations

from sagellm_benchmark.cli import (
    _prefer_newer_entry,
    build_canonical_path,
    build_idempotency_key,
)


def _entry(
    *,
    submitted_at: str,
    throughput_tps: float = 10.0,
    workload: str = "Q1",
    version: str = "0.5.1.2",
) -> dict:
    return {
        "entry_id": "test-entry",
        "sagellm_version": version,
        "config_type": "single_gpu",
        "workload": {"name": workload},
        "model": {"name": "sshleifer/tiny-gpt2", "precision": "fp32"},
        "hardware": {"chip_model": "cpu", "chip_count": 1},
        "cluster": {"node_count": 1},
        "metrics": {"throughput_tps": throughput_tps},
        "metadata": {"submitted_at": submitted_at, "release_date": "2026-02-20"},
        "versions": {"benchmark": version},
    }


def test_build_idempotency_key_stable_for_same_dimensions() -> None:
    """Idempotency key should be stable when key dimensions are same."""
    first = _entry(submitted_at="2026-02-20T10:00:00Z")
    second = _entry(submitted_at="2026-02-20T12:00:00Z", throughput_tps=99.0)

    assert build_idempotency_key(first) == build_idempotency_key(second)


def test_build_idempotency_key_changes_with_workload() -> None:
    """Different workload should produce different idempotency keys."""
    q1 = _entry(submitted_at="2026-02-20T10:00:00Z", workload="Q1")
    q2 = _entry(submitted_at="2026-02-20T10:00:00Z", workload="Q2")

    assert build_idempotency_key(q1) != build_idempotency_key(q2)


def test_prefer_newer_entry_uses_submitted_at_first() -> None:
    """Newer submitted_at should win when keys are the same."""
    old = _entry(submitted_at="2026-02-20T10:00:00Z", throughput_tps=999.0)
    new = _entry(submitted_at="2026-02-20T12:00:00Z", throughput_tps=1.0)

    preferred = _prefer_newer_entry(old, new)

    assert preferred is new


def test_prefer_newer_entry_falls_back_to_throughput_when_timestamps_equal() -> None:
    """Throughput should break ties when timestamps are equal."""
    left = _entry(submitted_at="2026-02-20T10:00:00Z", throughput_tps=20.0)
    right = _entry(submitted_at="2026-02-20T10:00:00Z", throughput_tps=30.0)

    preferred = _prefer_newer_entry(left, right)

    assert preferred is right


def test_build_canonical_path_is_deterministic() -> None:
    """Canonical path should be deterministic for same entry dimensions."""
    first = _entry(submitted_at="2026-02-20T10:00:00Z")
    second = _entry(submitted_at="2026-02-20T11:00:00Z")

    assert build_canonical_path(first) == build_canonical_path(second)
