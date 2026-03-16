"""Tests for performance CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from sagellm_protocol import Metrics, Timestamps

from sagellm_benchmark.cli import (
    _capture_target_runtime_artifacts,
    _display_perf_e2e_table,
    _display_results,
    _format_e2e_markdown,
    console,
    main,
)
from sagellm_benchmark.compare_runner import run_stream_compare_target
from sagellm_benchmark.nonstream_compare import (
    NonStreamCompareConfig,
    NonStreamTarget,
    run_nonstream_compare,
)
from sagellm_benchmark.types import AggregatedMetrics, BenchmarkResult


def test_perf_help():
    runner = CliRunner()
    result = runner.invoke(main, ["perf", "--help"])
    assert result.exit_code == 0
    assert "--type" in result.output
    assert "operator" in result.output
    assert "e2e" in result.output
    assert "--plot" in result.output
    assert "--plot-format" in result.output
    assert "single-endpoint" in result.output
    assert "cross-engine live benchmarks" in result.output


def test_perf_live_rejects_multiple_models():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "perf",
            "--type",
            "e2e",
            "--live",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--model",
            "meta-llama/Llama-3.2-1B-Instruct",
        ],
    )
    assert result.exit_code != 0
    assert "perf --live accepts exactly one --model" in result.output


def test_stream_compare_runner_uses_chat_streaming_client_only(monkeypatch) -> None:
    captured_endpoint_types: list[str] = []

    class FakeGatewayClient:
        def __init__(self, *, endpoint_type: str, **kwargs) -> None:
            del kwargs
            captured_endpoint_types.append(endpoint_type)

        async def health_check(self, timeout: float = 5.0) -> bool:
            del timeout
            return True

        async def discover_model(self, timeout: float = 5.0) -> str | None:
            del timeout
            return "Qwen/Qwen2.5-0.5B-Instruct"

        async def generate_batch(self, requests, concurrent: bool = True):
            del concurrent
            assert requests, "chat streaming compare boundary broke: no requests were generated"
            assert all(request.stream is True for request in requests), (
                "chat streaming compare boundary broke: compare runner generated non-stream requests"
            )
            return [
                BenchmarkResult(
                    request_id=request.request_id,
                    success=True,
                    error=None,
                    metrics=Metrics(
                        ttft_ms=10.0,
                        tbt_ms=2.0,
                        tpot_ms=3.0,
                        throughput_tps=100.0,
                        peak_mem_mb=0,
                        error_rate=0.0,
                        timestamps=Timestamps(
                            queued_at=1.0,
                            scheduled_at=1.0,
                            executed_at=1.01,
                            completed_at=1.05,
                        ),
                        itl_list=[2.0, 2.0],
                    ),
                    output_text="ok",
                    output_tokens=4,
                    prompt_tokens=16,
                    itl_list=[2.0, 2.0],
                    e2e_latency_ms=50.0,
                )
                for request in requests
            ]

        async def close(self) -> None:
            return None

    monkeypatch.setattr("sagellm_benchmark.compare_runner.GatewayClient", FakeGatewayClient)

    rows, summary = run_stream_compare_target(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        backend_url="http://127.0.0.1:8901/v1",
        batch_sizes=(1,),
        api_key="sagellm-benchmark",
        request_timeout=30.0,
        server_wait_s=1.0,
        max_seq_len_override=512,
        max_output_tokens_override=32,
    )

    assert captured_endpoint_types == ["chat"], (
        "chat streaming compare boundary reopened: compare runner no longer constructs the "
        "GatewayClient with endpoint_type='chat'"
    )
    assert rows[0]["transport"] == "stream"
    assert summary["avg_ttft_ms"] == 10.0


def test_perf_e2e_generates_files():
    runner = CliRunner()
    with runner.isolated_filesystem():
        json_path = "out/perf.json"
        md_path = "out/perf.md"
        result = runner.invoke(
            main,
            [
                "perf",
                "--type",
                "e2e",
                "--model",
                "Qwen/Qwen2-7B-Instruct",
                "--batch-size",
                "1",
                "--precision",
                "fp16",
                "--output-json",
                json_path,
                "--output-markdown",
                md_path,
            ],
        )
        assert result.exit_code == 0

        with open(json_path) as f:
            payload = json.load(f)
        assert payload["kind"] == "e2e"
        assert len(payload["rows"]) > 0
        assert "precision" in payload["rows"][0]


def test_report_accepts_perf_json():
    runner = CliRunner()
    with runner.isolated_filesystem():
        path = "perf.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "kind": "operator",
                    "device": "cpu",
                    "comparisons": [
                        {
                            "optimized_name": "CustomLinear",
                            "baseline_time_ms": 10.0,
                            "optimized_time_ms": 5.0,
                            "speedup": 2.0,
                            "time_saved_ms": 5.0,
                            "time_saved_pct": 50.0,
                        }
                    ],
                },
                f,
            )

        result = runner.invoke(main, ["report", "--input", path, "--format", "markdown"])
        assert result.exit_code == 0
        assert "Operator Benchmark Report" in result.output


def test_compare_generates_files(monkeypatch):
    """Compare command should write per-target and summary artifacts."""

    def fake_run_stream_compare_target(**kwargs):
        backend_url = kwargs["backend_url"]
        if "8902" in backend_url:
            ttft = 10.0
            tbt = 2.0
            tps = 100.0
        else:
            ttft = 12.0
            tbt = 3.0
            tps = 90.0
        rows = [
            {
                "model": kwargs["model"],
                "effective_model": kwargs["model"],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": ttft,
                "tbt_ms": tbt,
                "tpot_ms": 5.0,
                "throughput_tps": tps,
                "output_throughput_tps": tps,
                "request_throughput_rps": 1.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "avg_e2el_ms": 20.0,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ]
        return rows, {
            "total_rows": 1,
            "avg_ttft_ms": ttft,
            "avg_tbt_ms": tbt,
            "avg_tpot_ms": 5.0,
            "avg_itl_ms": 1.5,
            "avg_e2el_ms": 20.0,
            "avg_throughput_tps": tps,
            "avg_output_throughput_tps": tps,
            "avg_request_throughput_rps": 1.0,
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli.run_stream_compare_target",
        fake_run_stream_compare_target,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._capture_target_runtime_artifacts",
        lambda **kwargs: {
            "info_json": "dummy-info.json",
            "core_telemetry_json": "dummy-telemetry.json",
        },
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._validate_sagellm_explicit_decode_runtime",
        lambda **kwargs: None,
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("compare_out")
        result = runner.invoke(
            main,
            [
                "compare",
                "--target",
                "sagellm=http://127.0.0.1:8902/v1",
                "--target",
                "vllm=http://127.0.0.1:8901/v1",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--hardware-family",
                "cuda",
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "sagellm.json").exists()
        assert (output_dir / "sagellm.canonical.json").exists()
        assert (output_dir / "sagellm.parity.json").exists()
        assert (output_dir / "sagellm_leaderboard.json").exists()
        assert (output_dir / "vllm.json").exists()
        assert (output_dir / "vllm.canonical.json").exists()
        assert (output_dir / "vllm.parity.json").exists()
        assert (output_dir / "vllm_leaderboard.json").exists()
        assert (output_dir / "comparison.json").exists()
        assert (output_dir / "comparison.canonical.json").exists()
        assert (output_dir / "leaderboard_manifest.json").exists()
        assert (output_dir / "publish" / "website-ready" / "leaderboard_single.json").exists()
        assert (output_dir / "publish" / "website-ready" / "leaderboard_multi.json").exists()
        assert (output_dir / "publish" / "website-ready" / "leaderboard_compare.json").exists()
        assert (output_dir / "publish" / "website-ready" / "last_updated.json").exists()

        with open(output_dir / "sagellm.parity.json") as f:
            parity_payload = json.load(f)
        assert parity_payload["schema_version"] == "parity-run/v1"
        assert parity_payload["hardware_family"] == "cuda"
        assert parity_payload["scenarios"][0]["has_step_evidence"] is False

        with open(output_dir / "sagellm.canonical.json") as f:
            canonical_payload = json.load(f)
        assert canonical_payload["schema_version"] == "canonical-benchmark-result/v2"
        assert canonical_payload["artifact_kind"] == "execution_result"
        assert canonical_payload["artifacts"]["leaderboard_json"].endswith(
            "sagellm_leaderboard.json"
        )

        with open(output_dir / "sagellm_leaderboard.json") as f:
            leaderboard_payload = json.load(f)
        assert leaderboard_payload["schema_version"] == "leaderboard-export-entry/v2"
        assert leaderboard_payload["engine"] == "sagellm"
        assert leaderboard_payload["metadata"]["hardware_family"] == "cuda"
        assert leaderboard_payload["metadata"]["idempotency_key"]

        with open(output_dir / "comparison.json") as f:
            payload = json.load(f)
        assert payload["kind"] == "compare"
        assert payload["baseline"] == "sagellm"
        assert len(payload["targets"]) == 2
        assert json.loads((output_dir / "sagellm.json").read_text())["transport"] == "stream"
        manifest = json.loads((output_dir / "leaderboard_manifest.json").read_text())
        assert manifest["schema_version"] == "leaderboard-export-manifest/v2"
        assert len(manifest["entries"]) == 2
        compare_snapshot = json.loads(
            (output_dir / "publish" / "website-ready" / "leaderboard_compare.json").read_text()
        )
        assert compare_snapshot["schema_version"] == "leaderboard-compare-snapshot/v1"
        assert compare_snapshot["group_count"] >= 1


def test_compare_forwards_dataset_backed_stream_options(monkeypatch):
    captured_kwargs: list[dict[str, object]] = []

    def fake_run_stream_compare_target(**kwargs):
        captured_kwargs.append(kwargs)
        return [
            {
                "model": kwargs["model"],
                "effective_model": kwargs["model"],
                "precision": "live",
                "scenario": "random_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "tpot_ms": 5.0,
                "throughput_tps": 100.0,
                "output_throughput_tps": 100.0,
                "request_throughput_rps": 1.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "avg_e2el_ms": 20.0,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ], {
            "total_rows": 1,
            "avg_ttft_ms": 10.0,
            "avg_tbt_ms": 2.0,
            "avg_tpot_ms": 5.0,
            "avg_itl_ms": 1.5,
            "avg_e2el_ms": 20.0,
            "avg_throughput_tps": 100.0,
            "avg_output_throughput_tps": 100.0,
            "avg_request_throughput_rps": 1.0,
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli.run_stream_compare_target",
        fake_run_stream_compare_target,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._capture_target_runtime_artifacts",
        lambda **kwargs: {
            "info_json": "dummy-info.json",
            "core_telemetry_json": "dummy-telemetry.json",
        },
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._validate_sagellm_explicit_decode_runtime",
        lambda **kwargs: None,
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("compare_random_out")
        result = runner.invoke(
            main,
            [
                "compare",
                "--target",
                "sagellm=http://127.0.0.1:8902/v1",
                "--target",
                "vllm=http://127.0.0.1:8901/v1",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--hardware-family",
                "cuda",
                "--dataset-name",
                "random",
                "--num-prompts",
                "6",
                "--input-len",
                "512",
                "--output-len",
                "48",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0
    assert len(captured_kwargs) == 2
    assert all(kwargs["dataset_name"] == "random" for kwargs in captured_kwargs)
    assert all(kwargs["num_prompts"] == 6 for kwargs in captured_kwargs)
    assert all(kwargs["input_len"] == 512 for kwargs in captured_kwargs)
    assert all(kwargs["output_len"] == 48 for kwargs in captured_kwargs)


def test_compare_record_forwards_dataset_backed_stream_options(monkeypatch):
    captured_kwargs: list[dict[str, object]] = []

    def fake_run_stream_compare_target(**kwargs):
        captured_kwargs.append(kwargs)
        return [
            {
                "model": kwargs["model"],
                "effective_model": kwargs["model"],
                "precision": "live",
                "scenario": "sharegpt_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "tpot_ms": 5.0,
                "throughput_tps": 100.0,
                "output_throughput_tps": 100.0,
                "request_throughput_rps": 1.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "avg_e2el_ms": 20.0,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ], {
            "total_rows": 1,
            "avg_ttft_ms": 10.0,
            "avg_tbt_ms": 2.0,
            "avg_tpot_ms": 5.0,
            "avg_itl_ms": 1.5,
            "avg_e2el_ms": 20.0,
            "avg_throughput_tps": 100.0,
            "avg_output_throughput_tps": 100.0,
            "avg_request_throughput_rps": 1.0,
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli.run_stream_compare_target",
        fake_run_stream_compare_target,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._capture_target_runtime_artifacts",
        lambda **kwargs: {
            "info_json": "dummy-info.json",
            "core_telemetry_json": "dummy-telemetry.json",
        },
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._validate_sagellm_explicit_decode_runtime",
        lambda **kwargs: None,
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            [
                "compare-record",
                "--label",
                "sagellm",
                "--url",
                "http://127.0.0.1:8901/v1",
                "--hardware-family",
                "cuda",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--dataset-name",
                "sharegpt",
                "--num-prompts",
                "5",
                "--input-len",
                "256",
                "--output-len",
                "32",
                "--output-dir",
                "capture_out",
            ],
        )

    assert result.exit_code == 0
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["dataset_name"] == "sharegpt"
    assert captured_kwargs[0]["num_prompts"] == 5
    assert captured_kwargs[0]["input_len"] == 256
    assert captured_kwargs[0]["output_len"] == 32


def test_parity_gate_convert_core_telemetry_writes_normalized_artifact() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        input_path = Path("core_info.json")
        output_path = Path("telemetry/core_telemetry.json")
        input_path.write_text(
            json.dumps(
                {
                    "performance_mainline": {
                        "explicit_decode": {
                            "feature_gate": {
                                "feature_id": "runtime.native_decode.v1",
                                "default_enabled": False,
                                "enabled": True,
                                "rollout_state": "on",
                                "kill_switch_active": False,
                            },
                            "step_telemetry_schema_version": 1,
                            "step_telemetry_stable_fields": [
                                "trace_id",
                                "request_id",
                                "orchestration_step_id",
                                "batch_id",
                                "batch_type",
                                "step_index",
                                "batch_size",
                                "active_sequences",
                                "emitted_tokens",
                                "step_latency_ms",
                                "selected_implementation",
                                "selected_operator_pack",
                                "selection_interface_name",
                                "telemetry_source",
                            ],
                            "step_telemetry": [
                                {
                                    "trace_id": "trace-1",
                                    "request_id": "req-1",
                                    "orchestration_step_id": 3,
                                    "batch_id": 7,
                                    "batch_type": "decode",
                                    "step_index": 0,
                                    "batch_size": 1,
                                    "active_sequences": 1,
                                    "emitted_tokens": 1,
                                    "step_latency_ms": 0.6,
                                    "selected_implementation": "torch-fallback",
                                    "selected_operator_pack": "attention.decode",
                                    "selection_interface_name": "attention_decode",
                                    "telemetry_source": "step_trace",
                                }
                            ],
                            "step_telemetry_entries": 1,
                            "last_orchestration_step_id": 3,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            main,
            [
                "parity-gate",
                "convert-core-telemetry",
                "--input-json",
                str(input_path),
                "--label",
                "sagellm-after",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--hardware-family",
                "cuda",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "core-decode-step-telemetry/v1"
        assert payload["label"] == "sagellm-after"
        assert payload["step_telemetry_entries"] == 1
        assert payload["summary"]["step_records"] == 1


def test_nonstream_compare_module_generates_files():
    """Module runner should emit reusable non-stream compare artifacts."""

    responses = {
        "sagellm": {
            "ok": True,
            "status_code": 200,
            "elapsed_ms": 25.0,
            "completion_text": "sage reply",
            "finish_reason": "stop",
            "prompt_tokens": 16,
            "completion_tokens": 6,
            "total_tokens": 22,
            "raw_response": {},
        },
        "vllm": {
            "ok": True,
            "status_code": 200,
            "elapsed_ms": 15.0,
            "completion_text": "vllm reply",
            "finish_reason": "stop",
            "prompt_tokens": 16,
            "completion_tokens": 8,
            "total_tokens": 24,
            "raw_response": {},
        },
    }

    def fake_request(target, request_config):
        return dict(responses[target.label])

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = run_nonstream_compare(
            NonStreamCompareConfig(
                targets=(
                    NonStreamTarget("sagellm", "http://127.0.0.1:8901/v1"),
                    NonStreamTarget("vllm", "http://127.0.0.1:8000/v1"),
                ),
                model="Qwen/Qwen2.5-0.5B-Instruct",
                prompt="hello",
                batch_sizes=(1, 2),
                warmup_rounds=1,
                rounds=1,
                max_tokens=8,
                temperature=0.0,
                api_key="token",
                request_timeout=10.0,
                output_dir="nonstream_out",
            ),
            request_fn=fake_request,
        )

        assert output_dir == Path("nonstream_out")
        assert (output_dir / "sagellm.json").exists()
        assert (output_dir / "vllm.json").exists()
        assert (output_dir / "comparison.json").exists()
        assert (output_dir / "comparison.md").exists()

        with open(output_dir / "comparison.json") as f:
            payload = json.load(f)
        assert payload["kind"] == "nonstream_compare"
        assert payload["baseline"] == "sagellm"
        assert [target["label"] for target in payload["targets"]] == ["sagellm", "vllm"]


def test_nonstream_compare_preserves_request_metrics() -> None:
    """Non-stream compare should keep request-level latency/token compatibility fields."""

    def fake_request(target, request_config):
        del request_config
        if target.label == "sagellm":
            return {
                "ok": True,
                "status_code": 200,
                "elapsed_ms": 24.0,
                "completion_text": "sage reply",
                "finish_reason": "stop",
                "prompt_tokens": 16,
                "completion_tokens": 6,
                "total_tokens": 22,
                "raw_response": {"id": "sage-1"},
            }
        return {
            "ok": True,
            "status_code": 200,
            "elapsed_ms": 12.0,
            "completion_text": "vllm reply",
            "finish_reason": "stop",
            "prompt_tokens": 16,
            "completion_tokens": 4,
            "total_tokens": 20,
            "raw_response": {"id": "vllm-1"},
        }

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = run_nonstream_compare(
            NonStreamCompareConfig(
                targets=(
                    NonStreamTarget("sagellm", "http://127.0.0.1:8901/v1"),
                    NonStreamTarget("vllm", "http://127.0.0.1:8000/v1"),
                ),
                model="Qwen/Qwen2.5-0.5B-Instruct",
                prompt="hello",
                batch_sizes=(1,),
                warmup_rounds=0,
                rounds=1,
                max_tokens=8,
                temperature=0.0,
                api_key="token",
                request_timeout=10.0,
                output_dir="nonstream_metrics_out",
            ),
            request_fn=fake_request,
        )

        sagellm_payload = json.loads((output_dir / "sagellm.json").read_text(encoding="utf-8"))
        request = sagellm_payload["batches"][0]["requests"][0]

        assert request["ttft_ms"] == 24.0
        assert request["tpot_ms"] == 4.0
        assert request["e2e_latency_ms"] == 24.0
        assert request["throughput_tps"] == 250.0
        assert request["raw_response"]["id"] == "sage-1"
        assert sagellm_payload["summary"]["throughput_rps"] > 0


def test_compare_record_and_compare_offline_preserve_stream_metrics(monkeypatch) -> None:
    """compare-record/offline should preserve TTFT, ITL, TPOT, and E2EL fields."""

    def fake_run_stream_compare_target(**kwargs):
        backend_url = kwargs["backend_url"]
        if "8901" in backend_url:
            ttft = 11.0
            tbt = 2.5
            tpot = 4.5
            itl = 1.8
            e2el = 24.0
            tps = 88.0
            rps = 1.2
        else:
            ttft = 9.0
            tbt = 2.0
            tpot = 4.0
            itl = 1.2
            e2el = 19.0
            tps = 96.0
            rps = 1.4
        rows = [
            {
                "model": kwargs["model"],
                "effective_model": kwargs["model"],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": ttft,
                "tbt_ms": tbt,
                "tpot_ms": tpot,
                "avg_itl_ms": itl,
                "p50_itl_ms": itl,
                "p95_itl_ms": itl + 0.3,
                "p99_itl_ms": itl + 0.5,
                "throughput_tps": tps,
                "output_throughput_tps": tps,
                "request_throughput_rps": rps,
                "latency_p50_ms": e2el,
                "latency_p95_ms": e2el + 3.0,
                "latency_p99_ms": e2el + 5.0,
                "avg_e2el_ms": e2el,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ]
        return rows, {
            "total_rows": 1,
            "avg_ttft_ms": ttft,
            "avg_tbt_ms": tbt,
            "avg_tpot_ms": tpot,
            "avg_itl_ms": itl,
            "p50_itl_ms": itl,
            "p95_itl_ms": itl + 0.3,
            "p99_itl_ms": itl + 0.5,
            "avg_e2el_ms": e2el,
            "avg_throughput_tps": tps,
            "avg_output_throughput_tps": tps,
            "avg_request_throughput_rps": rps,
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli.run_stream_compare_target",
        fake_run_stream_compare_target,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._capture_target_runtime_artifacts",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._validate_sagellm_explicit_decode_runtime",
        lambda **kwargs: None,
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        sagellm_dir = Path("capture_sagellm")
        vllm_dir = Path("capture_vllm")
        sagellm_result = runner.invoke(
            main,
            [
                "compare-record",
                "--label",
                "sagellm",
                "--url",
                "http://127.0.0.1:8901/v1",
                "--hardware-family",
                "cuda",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--output-dir",
                str(sagellm_dir),
            ],
        )
        assert sagellm_result.exit_code == 0

        vllm_result = runner.invoke(
            main,
            [
                "compare-record",
                "--label",
                "vllm",
                "--url",
                "http://127.0.0.1:8902/v1",
                "--hardware-family",
                "cuda",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--output-dir",
                str(vllm_dir),
            ],
        )
        assert vllm_result.exit_code == 0

        sagellm_payload = json.loads((sagellm_dir / "sagellm.json").read_text(encoding="utf-8"))
        sagellm_canonical = json.loads(
            (sagellm_dir / "sagellm.canonical.json").read_text(encoding="utf-8")
        )
        sagellm_leaderboard = json.loads(
            (sagellm_dir / "sagellm_leaderboard.json").read_text(encoding="utf-8")
        )

        assert sagellm_payload["summary"]["avg_ttft_ms"] == 11.0
        assert sagellm_payload["summary"]["avg_itl_ms"] == 1.8
        assert sagellm_payload["summary"]["avg_tpot_ms"] == 4.5
        assert sagellm_payload["summary"]["avg_e2el_ms"] == 24.0
        assert sagellm_canonical["metrics"]["itl_ms"] == 1.8
        assert sagellm_canonical["metrics"]["e2el_ms"] == 24.0
        assert sagellm_leaderboard["metrics"]["ttft_ms"] == 11.0
        assert sagellm_leaderboard["metrics"]["itl_ms"] == 1.8
        assert sagellm_leaderboard["metrics"]["tpot_ms"] == 4.5
        assert sagellm_leaderboard["metrics"]["e2el_ms"] == 24.0

        offline_dir = Path("offline_compare")
        offline_result = runner.invoke(
            main,
            [
                "compare-offline",
                "--result",
                f"sagellm={sagellm_dir / 'sagellm.json'}",
                "--result",
                f"vllm={vllm_dir / 'vllm.json'}",
                "--output-dir",
                str(offline_dir),
            ],
        )

        assert offline_result.exit_code == 0
        comparison_payload = json.loads(
            (offline_dir / "comparison.json").read_text(encoding="utf-8")
        )
        comparison_canonical = json.loads(
            (offline_dir / "comparison.canonical.json").read_text(encoding="utf-8")
        )

        assert comparison_payload["targets"][0]["avg_itl_ms"] == 1.8
        assert comparison_payload["targets"][0]["avg_tpot_ms"] == 4.5
        assert comparison_payload["targets"][0]["avg_e2el_ms"] == 24.0
        assert comparison_payload["targets"][1]["avg_itl_ms"] == 1.2
        assert comparison_payload["targets"][1]["delta_vs_baseline"]["itl_ms"] == pytest.approx(
            -0.6
        )
        assert comparison_payload["targets"][1]["delta_vs_baseline"]["tpot_ms"] == pytest.approx(
            -0.5
        )
        assert comparison_payload["targets"][1]["delta_vs_baseline"]["e2el_ms"] == -5.0
        assert comparison_canonical["measurements"]["summary"]["targets"][0]["avg_itl_ms"] == 1.8
        assert comparison_canonical["measurements"]["summary"]["targets"][1]["delta_vs_baseline"][
            "itl_ms"
        ] == pytest.approx(-0.6)


def test_nonstream_compare_cli_invokes_module(monkeypatch):
    """CLI should forward parsed options into the reusable non-stream compare module."""

    captured: dict[str, object] = {}

    def fake_run_nonstream_compare(config):
        captured["config"] = config
        return Path("compare_out")

    monkeypatch.setattr("sagellm_benchmark.cli.run_nonstream_compare", fake_run_nonstream_compare)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "nonstream-compare",
            "--target",
            "sagellm=http://127.0.0.1:8901/v1",
            "--target",
            "vllm=http://127.0.0.1:8000/v1",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--prompt",
            "hello",
            "--batch-size",
            "1",
            "--batch-size",
            "2",
            "--rounds",
            "2",
            "--output-dir",
            "compare_out",
        ],
    )

    assert result.exit_code == 0
    config = captured["config"]
    assert isinstance(config, NonStreamCompareConfig)
    assert [target.label for target in config.targets] == ["sagellm", "vllm"]
    assert config.batch_sizes == (1, 2)
    assert config.rounds == 2
    assert config.output_dir == "compare_out"


def test_compare_record_generates_files(monkeypatch):
    """compare-record should write a single target payload for later offline compare."""

    def fake_run_stream_compare_target(**kwargs):
        rows = [
            {
                "model": kwargs["model"],
                "effective_model": kwargs["model"],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "tpot_ms": 5.0,
                "throughput_tps": 100.0,
                "output_throughput_tps": 100.0,
                "request_throughput_rps": 1.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "avg_e2el_ms": 20.0,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ]
        return rows, {
            "total_rows": 1,
            "avg_ttft_ms": 10.0,
            "avg_tbt_ms": 2.0,
            "avg_tpot_ms": 5.0,
            "avg_itl_ms": 1.5,
            "avg_e2el_ms": 20.0,
            "avg_throughput_tps": 100.0,
            "avg_output_throughput_tps": 100.0,
            "avg_request_throughput_rps": 1.0,
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli.run_stream_compare_target",
        fake_run_stream_compare_target,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._capture_target_runtime_artifacts",
        lambda **kwargs: {
            "info_json": "dummy-info.json",
            "core_telemetry_json": "dummy-telemetry.json",
        },
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._validate_sagellm_explicit_decode_runtime",
        lambda **kwargs: None,
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("capture_out")
        result = runner.invoke(
            main,
            [
                "compare-record",
                "--label",
                "sagellm",
                "--url",
                "http://127.0.0.1:8901/v1",
                "--hardware-family",
                "cuda",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "sagellm.json").exists()
        assert (output_dir / "sagellm.parity.json").exists()
        with open(output_dir / "sagellm.json") as f:
            payload = json.load(f)
        assert payload["kind"] == "e2e"
        assert payload["label"] == "sagellm"
        with open(output_dir / "sagellm.parity.json") as f:
            parity_payload = json.load(f)
        assert parity_payload["schema_version"] == "parity-run/v1"
        assert parity_payload["model"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_compare_record_auto_captures_core_telemetry_artifact(monkeypatch):
    """compare-record should persist runtime artifacts when /info exposes core telemetry."""

    def fake_run_stream_compare_target(**kwargs):
        rows = [
            {
                "model": kwargs["model"],
                "effective_model": kwargs["model"],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "tpot_ms": 5.0,
                "throughput_tps": 100.0,
                "output_throughput_tps": 100.0,
                "request_throughput_rps": 1.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "avg_e2el_ms": 20.0,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ]
        return rows, {
            "total_rows": 1,
            "avg_ttft_ms": 10.0,
            "avg_tbt_ms": 2.0,
            "avg_tpot_ms": 5.0,
            "avg_itl_ms": 1.5,
            "avg_e2el_ms": 20.0,
            "avg_throughput_tps": 100.0,
            "avg_output_throughput_tps": 100.0,
            "avg_request_throughput_rps": 1.0,
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli.run_stream_compare_target",
        fake_run_stream_compare_target,
    )

    def fake_capture_target_runtime_artifacts(**kwargs):
        output_dir = kwargs["output_dir"]
        info_path = output_dir / "sagellm_info.json"
        telemetry_path = output_dir / "sagellm_core_telemetry.json"
        info_path.write_text('{"engine":"sagellm"}\n', encoding="utf-8")
        telemetry_path.write_text(
            '{"schema_version":"core-decode-step-telemetry/v1"}\n', encoding="utf-8"
        )
        return {
            "info_json": str(info_path),
            "core_telemetry_json": str(telemetry_path),
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli._capture_target_runtime_artifacts",
        fake_capture_target_runtime_artifacts,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._validate_sagellm_explicit_decode_runtime",
        lambda **kwargs: None,
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("capture_out")
        result = runner.invoke(
            main,
            [
                "compare-record",
                "--label",
                "sagellm",
                "--url",
                "http://127.0.0.1:8901/v1",
                "--hardware-family",
                "cuda",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "sagellm_info.json").exists()
        assert (output_dir / "sagellm_core_telemetry.json").exists()
        payload = json.loads((output_dir / "sagellm.json").read_text())
        assert payload["runtime_artifacts"]["info_json"].endswith("sagellm_info.json")
        assert payload["runtime_artifacts"]["core_telemetry_json"].endswith(
            "sagellm_core_telemetry.json"
        )


def test_validate_serving_consistency_writes_report(monkeypatch):
    """validate-serving-consistency should persist a pass/fail report for one live target."""

    def fake_run_compare_target(**kwargs):
        output_dir = kwargs["output_dir"]
        info_path = output_dir / "sagellm_info.json"
        telemetry_path = output_dir / "sagellm_core_telemetry.json"
        info_path.write_text(
            json.dumps(
                {
                    "performance_mainline": {
                        "decode_runtime_diagnostics": {
                            "summary": {
                                "primary_decode_attention_hit": True,
                                "adjacent_decode_pack_hit": True,
                                "attention_selected_implementation": "native-cuda",
                                "attention_selected_operator_pack": "attention.decode.cuda.small_batch",
                                "attention_first_failure_reason": None,
                                "attention_batch_size": 1,
                                "attention_native_kernel_hit": True,
                                "attention_runtime_fallback": False,
                                "adjacent_selected_implementation": "native-cuda-small-batch-pack",
                                "adjacent_selected_operator_pack": "decode.step.small_batch",
                                "adjacent_native_kernel_hit": True,
                                "adjacent_runtime_fallback": False,
                            }
                        }
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        telemetry_path.write_text(
            json.dumps(
                {
                    "schema_version": "core-decode-step-telemetry/v1",
                    "summary": {
                        "batch_sizes": [1],
                        "selected_implementations": ["native-cuda-small-batch-pack"],
                        "selected_operator_packs": ["decode.step.small_batch"],
                        "by_batch_size": [
                            {
                                "batch_size": 1,
                                "step_records": 2,
                                "unique_requests": 2,
                                "avg_step_latency_ms": 0.6,
                                "max_step_latency_ms": 0.7,
                                "selected_implementations": ["native-cuda-small-batch-pack"],
                                "selected_operator_packs": ["decode.step.small_batch"],
                            }
                        ],
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "label": kwargs["label"],
            "url": kwargs["url"],
            "summary": {"avg_ttft_ms": 10.0, "avg_tbt_ms": 2.0, "avg_throughput_tps": 100.0},
            "json": str(output_dir / "sagellm.json"),
            "markdown": str(output_dir / "sagellm.md"),
            "parity_json": str(output_dir / "sagellm.parity.json"),
            "runtime_artifacts": {
                "info_json": str(info_path),
                "core_telemetry_json": str(telemetry_path),
            },
            "payload": {
                "rows": [
                    {
                        "batch_size": 1,
                        "successful_requests": 1,
                        "failed_requests": 0,
                    }
                ]
            },
        }

    monkeypatch.setattr("sagellm_benchmark.cli._run_compare_target", fake_run_compare_target)

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("validate_out")
        reference_path = Path("reference.json")
        reference_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "1": {
                            "attention_impl": "native-cuda",
                            "selected_pack": "decode.step.small_batch",
                        }
                    },
                    "raw": {
                        "composite_step": {
                            "1": {
                                "after": {
                                    "decode_runtime_diagnostics": {
                                        "summary": {
                                            "attention_selected_operator_pack": "attention.decode.cuda.small_batch",
                                            "adjacent_selected_implementation": "native-cuda-small-batch-pack",
                                            "adjacent_selected_operator_pack": "decode.step.small_batch",
                                        }
                                    }
                                }
                            }
                        }
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            main,
            [
                "validate-serving-consistency",
                "--label",
                "sagellm",
                "--url",
                "http://127.0.0.1:8901/v1",
                "--hardware-family",
                "cuda",
                "--reference-artifact",
                str(reference_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        report_path = output_dir / "sagellm_runtime_consistency.json"
        assert report_path.exists()
        payload = json.loads(report_path.read_text())
        assert payload["passed"] is True
        assert payload["observed_batch_size"] == 1


def test_capture_target_runtime_artifacts_supports_gateway_info_wrapper(monkeypatch, tmp_path):
    """Gateway /info should still yield a core telemetry artifact when one engine is registered."""

    monkeypatch.setattr(
        "sagellm_benchmark.cli._fetch_json_probe",
        lambda *args, **kwargs: {
            "service": "sageLLM Gateway",
            "registered_engines": [
                {
                    "engine_id": "engine-cuda-8902",
                    "info": {
                        "performance_mainline": {
                            "explicit_decode": {
                                "feature_gate": {
                                    "feature_id": "runtime.native_decode.v1",
                                    "default_enabled": False,
                                    "enabled": True,
                                    "rollout_state": "on",
                                    "kill_switch_active": False,
                                },
                                "step_telemetry_schema_version": 1,
                                "step_telemetry_stable_fields": [
                                    "trace_id",
                                    "request_id",
                                    "orchestration_step_id",
                                    "batch_id",
                                    "batch_type",
                                    "step_index",
                                    "batch_size",
                                    "active_sequences",
                                    "emitted_tokens",
                                    "step_latency_ms",
                                    "selected_implementation",
                                    "selected_operator_pack",
                                    "selection_interface_name",
                                    "telemetry_source",
                                ],
                                "step_telemetry": [
                                    {
                                        "trace_id": "trace-1",
                                        "request_id": "req-1",
                                        "orchestration_step_id": 1,
                                        "batch_id": 1,
                                        "batch_type": "decode",
                                        "step_index": 0,
                                        "batch_size": 1,
                                        "active_sequences": 1,
                                        "emitted_tokens": 1,
                                        "step_latency_ms": 0.5,
                                        "selected_implementation": "native-cuda-small-batch-pack",
                                        "selected_operator_pack": "decode.step.small_batch",
                                        "selection_interface_name": "decode_operator_pack",
                                        "telemetry_source": "runtime_counter",
                                    }
                                ],
                                "step_telemetry_entries": 1,
                                "last_orchestration_step_id": 1,
                            }
                        }
                    },
                }
            ],
        },
    )

    runtime_artifacts = _capture_target_runtime_artifacts(
        label="sagellm",
        url="http://127.0.0.1:8901/v1",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_family="cuda",
        api_key="sagellm-benchmark",
        request_timeout=30.0,
        output_dir=tmp_path,
    )

    assert runtime_artifacts["info_json"].endswith("sagellm_info.json")
    assert runtime_artifacts["core_telemetry_json"].endswith("sagellm_core_telemetry.json")


def test_compare_auto_captures_runtime_artifacts(monkeypatch):
    """compare should include per-target runtime artifact references in target payloads."""

    def fake_run_stream_compare_target(**kwargs):
        rows = [
            {
                "model": kwargs["model"],
                "effective_model": kwargs["model"],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "tpot_ms": 5.0,
                "throughput_tps": 100.0,
                "output_throughput_tps": 100.0,
                "request_throughput_rps": 1.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "avg_e2el_ms": 20.0,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ]
        return rows, {
            "total_rows": 1,
            "avg_ttft_ms": 10.0,
            "avg_tbt_ms": 2.0,
            "avg_tpot_ms": 5.0,
            "avg_itl_ms": 1.5,
            "avg_e2el_ms": 20.0,
            "avg_throughput_tps": 100.0,
            "avg_output_throughput_tps": 100.0,
            "avg_request_throughput_rps": 1.0,
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli.run_stream_compare_target",
        fake_run_stream_compare_target,
    )

    def fake_capture_target_runtime_artifacts(**kwargs):
        output_dir = kwargs["output_dir"]
        label = kwargs["label"]
        info_path = output_dir / f"{label}_info.json"
        info_path.write_text(json.dumps({"label": label}) + "\n", encoding="utf-8")
        return {"info_json": str(info_path)}

    monkeypatch.setattr(
        "sagellm_benchmark.cli._capture_target_runtime_artifacts",
        fake_capture_target_runtime_artifacts,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._validate_sagellm_explicit_decode_runtime",
        lambda **kwargs: None,
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("compare_out")
        result = runner.invoke(
            main,
            [
                "compare",
                "--target",
                "sagellm=http://127.0.0.1:8902/v1",
                "--target",
                "vllm=http://127.0.0.1:8901/v1",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--hardware-family",
                "cuda",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        sagellm_payload = json.loads((output_dir / "sagellm.json").read_text())
        vllm_payload = json.loads((output_dir / "vllm.json").read_text())
        assert sagellm_payload["runtime_artifacts"]["info_json"].endswith("sagellm_info.json")
        assert vllm_payload["runtime_artifacts"]["info_json"].endswith("vllm_info.json")


def test_compare_offline_generates_summary():
    """compare-offline should merge captured single-target results into comparison artifacts."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        input_dir = Path("captures")
        input_dir.mkdir()
        for label, ttft, tbt, tps in (
            ("sagellm", 10.0, 2.0, 100.0),
            ("vllm", 8.0, 1.0, 120.0),
        ):
            payload = {
                "kind": "e2e",
                "simulate": False,
                "mode": "live-compare",
                "label": label,
                "url": f"http://127.0.0.1/{label}/v1",
                "models": ["Qwen/Qwen2.5-0.5B-Instruct"],
                "batch_sizes": [1, 2, 4],
                "precisions": ["live"],
                "summary": {
                    "total_rows": 1,
                    "avg_ttft_ms": ttft,
                    "avg_tbt_ms": tbt,
                    "avg_throughput_tps": tps,
                },
                "rows": [],
            }
            with open(input_dir / f"{label}.json", "w") as f:
                json.dump(payload, f, indent=2)

        output_dir = Path("compare_out")
        result = runner.invoke(
            main,
            [
                "compare-offline",
                "--result",
                f"sagellm={input_dir / 'sagellm.json'}",
                "--result",
                f"vllm={input_dir / 'vllm.json'}",
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "comparison.json").exists()
        with open(output_dir / "comparison.json") as f:
            payload = json.load(f)
        assert payload["kind"] == "compare"
        assert payload["baseline"] == "sagellm"
        assert [target["label"] for target in payload["targets"]] == ["sagellm", "vllm"]


def test_compare_passes_target_commands(monkeypatch):
    """compare should forward optional target start commands to the execution layer."""

    captured: dict[str, object] = {}

    def fake_run_compare_command(**kwargs):
        captured.update(kwargs)
        return Path("compare_out")

    monkeypatch.setattr("sagellm_benchmark.cli._run_compare_command", fake_run_compare_command)
    monkeypatch.setattr(
        "sagellm_benchmark.cli._prepare_compare_publish_ready_outputs",
        lambda **kwargs: {},
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            "--target",
            "sagellm=http://127.0.0.1:8902/v1",
            "--target",
            "vllm=http://127.0.0.1:8000/v1",
            "--target-command",
            "sagellm=sagellm serve --port 8902",
            "--target-command",
            "vllm=vllm serve --port 8000",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--hardware-family",
            "cuda",
        ],
    )

    assert result.exit_code == 0
    assert captured["hardware_family"] == "cuda"
    assert captured["target_commands"] == {
        "sagellm": "sagellm serve --port 8902",
        "vllm": "vllm serve --port 8000",
    }


def test_display_results_emphasizes_output_throughput() -> None:
    metrics = AggregatedMetrics(
        avg_ttft_ms=12.5,
        avg_tbt_ms=4.0,
        avg_throughput_tps=80.0,
        output_throughput_tps=320.0,
        input_throughput_tps=40.0,
        request_throughput_rps=12.0,
        total_requests=16,
        successful_requests=16,
        failed_requests=0,
        peak_mem_mb=2048,
        total_input_tokens=400,
        total_output_tokens=3200,
    )

    with console.capture() as capture:
        _display_results({"sagellm": metrics})
    output = capture.get()

    assert "Output Throughput" in output
    assert "320.00 tokens/s" in output
    assert "Avg Per-Request TPS" in output
    assert "80.00 tokens/s" in output


def test_perf_e2e_summary_emphasizes_output_throughput() -> None:
    data = {
        "summary": {
            "total_rows": 2,
            "avg_ttft_ms": 12.0,
            "avg_tbt_ms": 3.0,
            "avg_throughput_tps": 75.0,
            "output_throughput_tps": 300.0,
        },
        "rows": [],
    }

    with console.capture() as capture:
        _display_perf_e2e_table(data)
    output = capture.get()

    assert "Output Throughput (tok/s): 300.00" in output
    assert "Avg Per-Request Throughput (tok/s): 75.00" in output

    markdown = _format_e2e_markdown(data)
    assert "- Output Throughput (tok/s): 300.00" in markdown
    assert "- Avg Per-Request Throughput (tok/s): 75.00" in markdown


def test_compare_prompt_cleanup_kills_local_targets(monkeypatch):
    """compare should offer to kill local target processes when requested."""

    def fake_run_stream_compare_target(**kwargs):
        rows = [
            {
                "model": kwargs["model"],
                "effective_model": kwargs["model"],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "tpot_ms": 5.0,
                "throughput_tps": 100.0,
                "output_throughput_tps": 100.0,
                "request_throughput_rps": 1.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "avg_e2el_ms": 20.0,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ]
        return rows, {
            "total_rows": 1,
            "avg_ttft_ms": 10.0,
            "avg_tbt_ms": 2.0,
            "avg_tpot_ms": 5.0,
            "avg_itl_ms": 1.5,
            "avg_e2el_ms": 20.0,
            "avg_throughput_tps": 100.0,
            "avg_output_throughput_tps": 100.0,
            "avg_request_throughput_rps": 1.0,
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli.run_stream_compare_target",
        fake_run_stream_compare_target,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._capture_target_runtime_artifacts",
        lambda **kwargs: {
            "info_json": "dummy-info.json",
            "core_telemetry_json": "dummy-telemetry.json",
        },
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._validate_sagellm_explicit_decode_runtime",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._discover_local_target_processes",
        lambda parsed_targets: [
            {
                "pid": 1234,
                "labels": ["sagellm"],
                "urls": ["http://127.0.0.1:8902/v1"],
                "ports": [8902],
                "command": "sagellm serve --port 8902",
            }
        ],
    )

    captured: dict[str, object] = {}

    def fake_terminate_processes(pids, *, grace_period_s=3.0):
        captured["pids"] = pids
        captured["grace_period_s"] = grace_period_s
        return {"terminated": [1234], "killed": [], "failed": []}

    monkeypatch.setattr("sagellm_benchmark.cli._terminate_processes", fake_terminate_processes)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            [
                "compare",
                "--target",
                "sagellm=http://127.0.0.1:8902/v1",
                "--target",
                "vllm=http://127.0.0.1:8901/v1",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--hardware-family",
                "cuda",
                "--prompt-cleanup",
            ],
            input="y\n",
        )

    assert result.exit_code == 0
    assert "Kill detected local target processes now?" in result.output
    assert captured["pids"] == [1234]
    assert "Cleanup complete" in result.output


def test_compare_prompt_cleanup_can_leave_targets_running(monkeypatch):
    """compare should respect a negative cleanup confirmation."""

    def fake_run_stream_compare_target(**kwargs):
        rows = [
            {
                "model": kwargs["model"],
                "effective_model": kwargs["model"],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "tpot_ms": 5.0,
                "throughput_tps": 100.0,
                "output_throughput_tps": 100.0,
                "request_throughput_rps": 1.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "avg_e2el_ms": 20.0,
                "memory_mb": 0.0,
                "mode": "live",
                "transport": "stream",
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ]
        return rows, {
            "total_rows": 1,
            "avg_ttft_ms": 10.0,
            "avg_tbt_ms": 2.0,
            "avg_tpot_ms": 5.0,
            "avg_itl_ms": 1.5,
            "avg_e2el_ms": 20.0,
            "avg_throughput_tps": 100.0,
            "avg_output_throughput_tps": 100.0,
            "avg_request_throughput_rps": 1.0,
        }

    monkeypatch.setattr(
        "sagellm_benchmark.cli.run_stream_compare_target",
        fake_run_stream_compare_target,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._capture_target_runtime_artifacts",
        lambda **kwargs: {
            "info_json": "dummy-info.json",
            "core_telemetry_json": "dummy-telemetry.json",
        },
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._validate_sagellm_explicit_decode_runtime",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.cli._discover_local_target_processes",
        lambda parsed_targets: [
            {
                "pid": 5678,
                "labels": ["vllm"],
                "urls": ["http://127.0.0.1:8000/v1"],
                "ports": [8000],
                "command": "vllm serve --port 8000",
            }
        ],
    )

    def fail_terminate_processes(pids, *, grace_period_s=3.0):
        raise AssertionError(f"terminate should not be called: {pids} {grace_period_s}")

    monkeypatch.setattr("sagellm_benchmark.cli._terminate_processes", fail_terminate_processes)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            [
                "compare",
                "--target",
                "sagellm=http://127.0.0.1:8902/v1",
                "--target",
                "vllm=http://127.0.0.1:8000/v1",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--hardware-family",
                "cuda",
                "--prompt-cleanup",
            ],
            input="n\n",
        )

    assert result.exit_code == 0
    assert "Leaving local benchmark target processes running." in result.output
