"""Tests for canonical artifact to leaderboard export."""

from __future__ import annotations

import json
from pathlib import Path

from sagellm_benchmark.canonical_artifacts import (
    build_live_compare_artifact,
    export_standard_leaderboard_artifacts,
    validate_canonical_artifact,
)
from sagellm_benchmark.exporters import LeaderboardExporter
from sagellm_benchmark.types import AggregatedMetrics


def _canonical_execution_result(
    *,
    workload_name: str = "Q1",
    engine: str = "sagellm",
    engine_version: str = "0.6.0.0",
    hardware_family: str = "cuda",
    chip_count: int = 1,
    node_count: int = 1,
    precision: str = "FP16",
    producer_command: str = "compare",
    workload_mode: str = "live-compare",
) -> dict:
    artifact = {
        "schema_version": "canonical-benchmark-result/v2",
        "artifact_kind": "execution_result",
        "artifact_id": "11111111-1111-1111-1111-111111111111",
        "producer": {"name": "sagellm-benchmark", "command": producer_command},
        "provenance": {
            "captured_at": "2026-03-14T12:00:00+00:00",
            "endpoint_url": "http://127.0.0.1:8901/v1",
            "output_dir": "benchmark_results/compare_test",
        },
        "hardware": {
            "family": hardware_family,
            "vendor": "NVIDIA" if hardware_family == "cuda" else "Huawei",
            "chip_model": "A100" if hardware_family == "cuda" else "Ascend 910B",
            "chip_count": chip_count,
            "chips_per_node": max(1, chip_count // node_count),
            "node_count": node_count,
            "interconnect": "NVLink" if hardware_family == "cuda" else "HCCS",
        },
        "engine": {
            "name": engine,
            "version": engine_version,
            "backend": hardware_family,
            "precision": precision,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "precision": precision,
        },
        "versions": {
            "benchmark": "0.6.0.0",
            "sagellm": "0.6.0.0",
            "vllm": "0.8.2",
        },
        "workload": {
            "name": workload_name,
            "mode": workload_mode,
            "precision": precision,
            "dataset": "default",
            "batch_size": 1,
            "concurrency": 1,
        },
        "metrics": {
            "ttft_ms": 10.0,
            "tbt_ms": 2.0,
            "tpot_ms": 3.0,
            "itl_ms": 1.5,
            "p50_itl_ms": 1.2,
            "p95_itl_ms": 2.3,
            "p99_itl_ms": 2.8,
            "e2el_ms": 18.0,
            "p50_e2el_ms": 17.0,
            "p95_e2el_ms": 24.0,
            "p99_e2el_ms": 28.0,
            "throughput_tps": 120.0,
            "request_throughput_rps": 4.0,
            "input_throughput_tps": 32.0,
            "output_throughput_tps": 120.0,
            "total_throughput_tps": 152.0,
            "total_input_tokens": 512,
            "total_output_tokens": 2048,
            "peak_mem_mb": 1024,
            "error_rate": 0.0,
            "prefix_hit_rate": 0.1,
            "kv_used_tokens": 1024,
            "kv_used_bytes": 4096,
            "evict_count": 0,
            "evict_ms": 0.0,
            "spec_accept_rate": 0.0,
        },
        "measurements": {"rows": [{"precision": precision, "batch_size": 1}]},
        "telemetry": {},
        "validation": {"publishable_to_leaderboard": True},
        "artifacts": {},
        "relations": [],
    }
    if node_count > 1:
        artifact["cluster"] = {
            "node_count": node_count,
            "comm_backend": "nccl" if hardware_family == "cuda" else "hccl",
            "topology_type": "multi_node",
            "parallelism": {
                "tensor_parallel": chip_count,
                "pipeline_parallel": 1,
                "data_parallel": 1,
            },
        }
    return artifact


def test_leaderboard_entry_from_canonical_q_workload() -> None:
    entry = LeaderboardExporter.leaderboard_entry_from_canonical_artifact(
        _canonical_execution_result(workload_name="Q3")
    )

    assert entry["schema_version"] == "leaderboard-export-entry/v2"
    assert entry["workload"]["name"] == "Q3"
    assert entry["workload"]["input_length"] == 128
    assert entry["workload"]["output_length"] == 256
    assert entry["engine"] == "sagellm"
    assert entry["model"]["precision"] == "FP16"
    assert entry["metrics"]["itl_ms"] == 1.5
    assert entry["metrics"]["e2el_ms"] == 18.0
    assert entry["metrics"]["request_throughput_rps"] == 4.0
    assert entry["metrics"]["output_throughput_tps"] == 120.0
    assert entry["metadata"]["idempotency_key"]
    assert entry["canonical_path"].endswith("_leaderboard.json")


def test_leaderboard_entry_from_canonical_multi_node_compare() -> None:
    entry = LeaderboardExporter.leaderboard_entry_from_canonical_artifact(
        _canonical_execution_result(
            workload_name="compare-live",
            engine="vllm-ascend",
            engine_version="0.11.0",
            hardware_family="ascend",
            chip_count=16,
            node_count=2,
            precision="BF16",
        )
    )

    assert entry["engine"] == "vllm-ascend"
    assert entry["engine_version"] == "0.11.0"
    assert entry["config_type"] == "multi_node"
    assert entry["cluster"]["node_count"] == 2
    assert entry["hardware"]["chip_count"] == 16
    assert entry["model"]["precision"] == "BF16"
    assert entry["metadata"]["hardware_family"] == "ascend"


def test_collect_entries_from_directory_prefers_canonical(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    compare_dir.mkdir()
    canonical_path = compare_dir / "sagellm.canonical.json"
    canonical_path.write_text(
        json.dumps(_canonical_execution_result(workload_name="Q5"), indent=2),
        encoding="utf-8",
    )
    leaderboard_path = compare_dir / "sagellm_leaderboard.json"
    entry = LeaderboardExporter.export_canonical_artifact(
        _canonical_execution_result(workload_name="Q5"),
        leaderboard_path,
    )
    LeaderboardExporter.register_exported_entry(
        output_dir=compare_dir,
        entry=entry,
        leaderboard_path=leaderboard_path,
        canonical_artifact_path=canonical_path,
    )

    entries, errors = LeaderboardExporter.collect_entries_from_directory(compare_dir)

    assert errors == []
    assert len(entries) == 1
    assert entries[0]["workload"]["name"] == "Q5"
    assert entries[0]["schema_version"] == "leaderboard-export-entry/v2"
    assert entries[0]["metadata"]["manifest_source"].endswith("leaderboard_manifest.json")


def test_export_canonical_artifact_end_to_end(tmp_path: Path) -> None:
    artifact = _canonical_execution_result(workload_name="Q8", engine="vllm")
    output_path = tmp_path / "vllm_leaderboard.json"

    entry = LeaderboardExporter.export_canonical_artifact(artifact, output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["entry_id"] == entry["entry_id"]
    assert saved["workload"]["name"] == "Q8"
    assert saved["workload"]["input_length"] == 192
    assert saved["workload"]["output_length"] == 128
    assert saved["engine"] == "vllm"
    assert saved["metadata"]["idempotency_key"] == entry["metadata"]["idempotency_key"]


def test_collect_entries_requires_manifest(tmp_path: Path) -> None:
    artifact = _canonical_execution_result(workload_name="Q2")
    (tmp_path / "Q2_leaderboard.json").write_text(json.dumps(artifact), encoding="utf-8")

    entries, errors = LeaderboardExporter.collect_entries_from_directory(tmp_path)

    assert entries == []
    assert errors
    assert "leaderboard_manifest.json" in errors[0]


def test_leaderboard_export_rejects_legacy_metric_aliases() -> None:
    artifact = _canonical_execution_result(workload_name="Q4")
    artifact["metrics"] = {
        "avg_ttft_ms": 10.0,
        "avg_tbt_ms": 2.0,
        "avg_tpot_ms": 3.0,
        "avg_itl_ms": 1.5,
        "avg_e2el_ms": 18.0,
        "avg_output_throughput_tps": 120.0,
        "avg_request_throughput_rps": 4.0,
        "peak_mem_mb": 1024,
        "error_rate": 0.0,
        "prefix_hit_rate": 0.1,
        "kv_used_tokens": 1024,
        "kv_used_bytes": 4096,
        "evict_count": 0,
        "evict_ms": 0.0,
    }

    try:
        LeaderboardExporter.leaderboard_entry_from_canonical_artifact(artifact)
    except ValueError as exc:
        assert str(exc) == "leaderboard entry.metrics.ttft_ms must be numeric"
    else:
        raise AssertionError(
            "legacy leaderboard metric aliases were accepted again; export must require ttft_ms/tbt_ms/tpot_ms/itl_ms/e2el_ms directly"
        )


def test_leaderboard_export_rejects_legacy_canonical_schema_version() -> None:
    artifact = _canonical_execution_result()
    artifact["schema_version"] = "canonical-benchmark-result/v1"

    try:
        LeaderboardExporter.leaderboard_entry_from_canonical_artifact(artifact)
    except ValueError as exc:
        assert (
            str(exc)
            == "leaderboard export requires canonical compare artifacts with schema_version 'canonical-benchmark-result/v2'"
        )
    else:
        raise AssertionError(
            "legacy canonical schema was accepted again; leaderboard export must stay pinned to canonical-benchmark-result/v2"
        )


def test_leaderboard_export_rejects_non_live_compare_artifact() -> None:
    artifact = _canonical_execution_result(producer_command="run", workload_mode="traffic")

    try:
        LeaderboardExporter.leaderboard_entry_from_canonical_artifact(artifact)
    except ValueError as exc:
        assert "canonical live compare artifacts" in str(exc)
    else:
        raise AssertionError("non-live-compare artifacts should be rejected")


def test_leaderboard_export_rejects_removed_direct_metrics_path(tmp_path: Path) -> None:
    try:
        LeaderboardExporter.export_to_leaderboard(
            AggregatedMetrics(
                avg_ttft_ms=10.0,
                avg_tbt_ms=2.0,
                avg_throughput_tps=80.0,
                total_throughput_tps=80.0,
            ),
            config={"model": "Qwen/Qwen2.5-0.5B-Instruct", "backend": "cuda"},
            workload_name="Q1",
            output_path=tmp_path / "Q1_leaderboard.json",
        )
    except ValueError as exc:
        assert (
            str(exc)
            == "Direct leaderboard export from AggregatedMetrics has been removed; export only from canonical live compare artifacts"
        )
    else:
        raise AssertionError(
            "removed direct leaderboard export path was reopened; raw AggregatedMetrics must never bypass canonical live compare artifacts"
        )


def test_live_compare_canonical_and_leaderboard_keep_stream_metrics_compatible() -> None:
    artifact = build_live_compare_artifact(
        label="sagellm",
        url="http://127.0.0.1:8901/v1",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_family="cuda",
        batch_sizes=[1, 2],
        summary={
            "total_rows": 2,
            "avg_ttft_ms": 10.0,
            "avg_tbt_ms": 2.0,
            "avg_tpot_ms": 3.5,
            "avg_itl_ms": 1.4,
            "p50_itl_ms": 1.2,
            "p95_itl_ms": 1.8,
            "p99_itl_ms": 2.1,
            "avg_e2el_ms": 18.0,
            "avg_throughput_tps": 80.0,
            "avg_output_throughput_tps": 84.0,
            "avg_request_throughput_rps": 4.0,
        },
        rows=[
            {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "effective_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "precision": "live",
                "scenario": "vllm_random_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "tpot_ms": 3.5,
                "avg_itl_ms": 1.4,
                "p50_itl_ms": 1.2,
                "p95_itl_ms": 1.8,
                "p99_itl_ms": 2.1,
                "throughput_tps": 80.0,
                "output_throughput_tps": 84.0,
                "request_throughput_rps": 4.0,
                "latency_p50_ms": 17.0,
                "latency_p95_ms": 21.0,
                "latency_p99_ms": 23.0,
                "avg_e2el_ms": 18.0,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ],
        runtime_artifacts={"info_json": "sagellm_info.json"},
        versions={"benchmark": "0.6.0.0", "sagellm": "0.6.0.0"},
        workload_context={
            "workload_profile": "vllm_random",
            "supplements": [],
            "dataset_name": "random",
            "scenario_source": "mainline",
        },
    )
    artifact["hardware"].update(
        {
            "vendor": "NVIDIA",
            "chip_model": "A100",
            "chip_count": 1,
            "chips_per_node": 1,
            "interconnect": "None",
            "node_count": 1,
        }
    )

    validate_canonical_artifact(artifact)
    assert artifact["schema_version"] == "canonical-benchmark-result/v2"
    entry = LeaderboardExporter.leaderboard_entry_from_canonical_artifact(artifact)
    LeaderboardExporter.validate_leaderboard_entry(entry)

    assert entry["metrics"]["ttft_ms"] == 10.0
    assert entry["metrics"]["tpot_ms"] == 3.5
    assert entry["metrics"]["itl_ms"] == 1.4
    assert entry["metrics"]["p95_itl_ms"] == 1.8
    assert entry["metrics"]["e2el_ms"] == 18.0
    assert entry["metrics"]["throughput_tps"] == 84.0
    assert entry["metrics"]["output_throughput_tps"] == 84.0


def test_build_compare_snapshot_prefers_sagellm_vs_vllm_pair() -> None:
    sagellm_entry = LeaderboardExporter.leaderboard_entry_from_canonical_artifact(
        _canonical_execution_result(
            workload_name="Q5",
            engine="sagellm",
            engine_version="0.6.0.0",
        )
    )
    vllm_artifact = _canonical_execution_result(
        workload_name="Q5",
        engine="vllm",
        engine_version="0.8.2",
    )
    vllm_artifact["metrics"]["throughput_tps"] = 100.0
    vllm_artifact["metrics"]["output_throughput_tps"] = 100.0
    vllm_artifact["metrics"]["ttft_ms"] = 12.0
    vllm_artifact["metrics"]["tbt_ms"] = 2.5
    vllm_entry = LeaderboardExporter.leaderboard_entry_from_canonical_artifact(vllm_artifact)

    snapshot = LeaderboardExporter.build_compare_snapshot([sagellm_entry, vllm_entry])

    assert snapshot["schema_version"] == "leaderboard-compare-snapshot/v1"
    assert snapshot["group_count"] == 1
    assert snapshot["preferred_pair_count"] == 1
    pair = snapshot["preferred_pairs"][0]["preferred_pair"]
    assert pair["left"]["engine"] == "sagellm"
    assert pair["right"]["engine"] == "vllm"
    assert pair["deltas"]["throughput_pct_left_vs_right"] == 20.0
    assert round(pair["deltas"]["ttft_pct_left_vs_right"], 4) == -16.6667
    assert pair["winners"]["throughput"] == "left"
    assert pair["winners"]["ttft"] == "left"


def test_export_standard_leaderboard_mainline_only_by_default(tmp_path: Path) -> None:
    artifact = build_live_compare_artifact(
        label="sagellm",
        url="http://127.0.0.1:8901/v1",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_family="cuda",
        batch_sizes=[1],
        summary={
            "total_rows": 2,
            "avg_ttft_ms": 20.0,
            "avg_tbt_ms": 4.0,
            "avg_tpot_ms": 6.0,
            "avg_itl_ms": 2.0,
            "avg_e2el_ms": 40.0,
            "avg_throughput_tps": 50.0,
            "avg_output_throughput_tps": 50.0,
            "avg_request_throughput_rps": 1.0,
        },
        rows=[
            {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "effective_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "precision": "live",
                "scenario": "vllm_random_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "tpot_ms": 3.0,
                "avg_itl_ms": 1.0,
                "throughput_tps": 100.0,
                "output_throughput_tps": 100.0,
                "request_throughput_rps": 2.0,
                "avg_e2el_ms": 20.0,
                "successful_requests": 1,
                "failed_requests": 0,
                "scenario_source": "mainline",
                "workload_profile": "vllm_random",
                "supplements": ["q1q8_supplement"],
                "dataset_name": "random",
            },
            {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "effective_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "precision": "live",
                "scenario": "q1_b1",
                "batch_size": 1,
                "ttft_ms": 30.0,
                "tbt_ms": 6.0,
                "tpot_ms": 9.0,
                "avg_itl_ms": 3.0,
                "throughput_tps": 20.0,
                "output_throughput_tps": 20.0,
                "request_throughput_rps": 0.5,
                "avg_e2el_ms": 60.0,
                "successful_requests": 1,
                "failed_requests": 0,
                "scenario_source": "supplement",
                "workload_profile": "vllm_random",
                "supplements": ["q1q8_supplement"],
                "dataset_name": "random",
            },
        ],
        runtime_artifacts={},
        versions={"benchmark": "0.6.0.0", "sagellm": "0.6.0.0"},
        workload_context={
            "workload_profile": "vllm_random",
            "supplements": ["q1q8_supplement"],
            "dataset_name": "random",
            "scenario_source": "mixed",
        },
    )
    canonical_path = tmp_path / "sagellm.canonical.json"
    canonical_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    export_standard_leaderboard_artifacts(tmp_path)
    default_entry = json.loads((tmp_path / "sagellm_leaderboard.json").read_text(encoding="utf-8"))
    assert default_entry["metrics"]["throughput_tps"] == 100.0

    for path in (tmp_path / "leaderboard_manifest.json", tmp_path / "sagellm_leaderboard.json"):
        if path.exists():
            path.unlink()

    export_standard_leaderboard_artifacts(tmp_path, include_supplements=True)
    include_entry = json.loads((tmp_path / "sagellm_leaderboard.json").read_text(encoding="utf-8"))
    assert include_entry["metrics"]["throughput_tps"] == 50.0
