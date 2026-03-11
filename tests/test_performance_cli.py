"""Tests for performance CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sagellm_benchmark.cli import main
from sagellm_benchmark.nonstream_compare import (
    NonStreamCompareConfig,
    NonStreamTarget,
    run_nonstream_compare,
)


def test_perf_help():
    runner = CliRunner()
    result = runner.invoke(main, ["perf", "--help"])
    assert result.exit_code == 0
    assert "--type" in result.output
    assert "operator" in result.output
    assert "e2e" in result.output
    assert "--plot" in result.output
    assert "--plot-format" in result.output


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

    def fake_run_e2e_model_benchmarks(**kwargs):
        backend_url = kwargs["backend_url"]
        if "8902" in backend_url:
            ttft = 10.0
            tbt = 2.0
            tps = 100.0
        else:
            ttft = 12.0
            tbt = 3.0
            tps = 90.0
        return [
            {
                "model": kwargs["models"][0],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": ttft,
                "tbt_ms": tbt,
                "throughput_tps": tps,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "memory_mb": 0.0,
                "mode": "live",
            }
        ]

    monkeypatch.setattr(
        "sagellm_benchmark.performance.model_benchmarks.run_e2e_model_benchmarks",
        fake_run_e2e_model_benchmarks,
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
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "sagellm.json").exists()
        assert (output_dir / "vllm.json").exists()
        assert (output_dir / "comparison.json").exists()

        with open(output_dir / "comparison.json") as f:
            payload = json.load(f)
        assert payload["kind"] == "compare"
        assert payload["baseline"] == "sagellm"
        assert len(payload["targets"]) == 2


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


def test_vllm_compare_run_generates_files(monkeypatch):
    """vllm-compare run should write the same compare artifacts with semantic labels."""

    def fake_run_e2e_model_benchmarks(**kwargs):
        backend_url = kwargs["backend_url"]
        if "8901" in backend_url:
            ttft = 11.0
            tbt = 2.5
            tps = 80.0
        else:
            ttft = 9.0
            tbt = 1.5
            tps = 95.0
        return [
            {
                "model": kwargs["models"][0],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": ttft,
                "tbt_ms": tbt,
                "throughput_tps": tps,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "memory_mb": 0.0,
                "mode": "live",
            }
        ]

    monkeypatch.setattr(
        "sagellm_benchmark.performance.model_benchmarks.run_e2e_model_benchmarks",
        fake_run_e2e_model_benchmarks,
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("compare_out")
        result = runner.invoke(
            main,
            [
                "vllm-compare",
                "run",
                "--sagellm-url",
                "http://127.0.0.1:8901/v1",
                "--vllm-url",
                "http://127.0.0.1:8000/v1",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "sagellm.json").exists()
        assert (output_dir / "vllm.json").exists()
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
        ],
    )

    assert result.exit_code == 0
    assert captured["target_commands"] == {
        "sagellm": "sagellm serve --port 8902",
        "vllm": "vllm serve --port 8000",
    }


def test_vllm_compare_run_passes_start_commands(monkeypatch):
    """vllm-compare run should map convenience start flags into target command wiring."""

    captured: dict[str, object] = {}

    def fake_run_compare_command(**kwargs):
        captured.update(kwargs)
        return Path("compare_out")

    monkeypatch.setattr("sagellm_benchmark.cli._run_compare_command", fake_run_compare_command)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "vllm-compare",
            "run",
            "--sagellm-url",
            "http://127.0.0.1:8901/v1",
            "--vllm-url",
            "http://127.0.0.1:8000/v1",
            "--start-sagellm-cmd",
            "sagellm serve --port 8901",
            "--start-vllm-cmd",
            "vllm serve --port 8000",
        ],
    )

    assert result.exit_code == 0
    assert captured["target_commands"] == {
        "sagellm": "sagellm serve --port 8901",
        "vllm": "vllm serve --port 8000",
    }


def test_compare_prompt_cleanup_kills_local_targets(monkeypatch):
    """compare should offer to kill local target processes when requested."""

    def fake_run_e2e_model_benchmarks(**kwargs):
        return [
            {
                "model": kwargs["models"][0],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "throughput_tps": 100.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "memory_mb": 0.0,
                "mode": "live",
            }
        ]

    monkeypatch.setattr(
        "sagellm_benchmark.performance.model_benchmarks.run_e2e_model_benchmarks",
        fake_run_e2e_model_benchmarks,
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

    def fake_run_e2e_model_benchmarks(**kwargs):
        return [
            {
                "model": kwargs["models"][0],
                "precision": "live",
                "scenario": "short_b1",
                "batch_size": 1,
                "ttft_ms": 10.0,
                "tbt_ms": 2.0,
                "throughput_tps": 100.0,
                "latency_p50_ms": 20.0,
                "latency_p95_ms": 25.0,
                "latency_p99_ms": 30.0,
                "memory_mb": 0.0,
                "mode": "live",
            }
        ]

    monkeypatch.setattr(
        "sagellm_benchmark.performance.model_benchmarks.run_e2e_model_benchmarks",
        fake_run_e2e_model_benchmarks,
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
                "--prompt-cleanup",
            ],
            input="n\n",
        )

    assert result.exit_code == 0
    assert "Leaving local benchmark target processes running." in result.output
