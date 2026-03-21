"""Tests for CLI functionality."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from sagellm_benchmark.cli import (
    _apply_compare_safe_env_defaults,
    _capture_target_runtime_artifacts,
    _validate_sagellm_explicit_decode_runtime,
    main,
)
from sagellm_benchmark.types import AggregatedMetrics


def test_cli_version():
    """Test CLI version command."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help():
    """Test CLI help command."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "sageLLM Benchmark Suite" in result.output
    assert "run" in result.output
    assert "compare" in result.output
    assert "nonstream-compare" in result.output
    assert "vllm-compare" in result.output
    assert "report" in result.output


def test_run_help():
    """Test run subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--profile" in result.output
    assert "--supplement" in result.output
    assert "--backend" in result.output
    assert "--model" in result.output
    assert "--output" in result.output


def test_report_help():
    """Test report subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["report", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--format" in result.output


def test_compare_help():
    """Test compare subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["compare", "--help"])
    assert result.exit_code == 0
    assert "--target" in result.output
    assert "--model" in result.output
    assert "--hardware-family" in result.output
    assert "--batch-size" in result.output
    assert "--profile" in result.output
    assert "--supplement" in result.output
    assert "--num-prompts" in result.output
    assert "--input-len" in result.output
    assert "--output-len" in result.output


def test_compare_record_help():
    """Test compare-record subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["compare-record", "--help"])
    assert result.exit_code == 0
    assert "--label" in result.output
    assert "--url" in result.output
    assert "--hardware-family" in result.output
    assert "--profile" in result.output


def test_validate_serving_consistency_help():
    """Test validate-serving-consistency subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["validate-serving-consistency", "--help"])
    assert result.exit_code == 0
    assert "--reference-artifact" in result.output
    assert "--batch-size" in result.output
    assert "--hardware-family" in result.output


def test_compare_offline_help():
    """Test compare-offline subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["compare-offline", "--help"])
    assert result.exit_code == 0
    assert "--result" in result.output


def test_compare_requires_multiple_targets():
    """Compare should require at least two targets."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            "--target",
            "sagellm=http://127.0.0.1:8000/v1",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--hardware-family",
            "cuda",
        ],
    )
    assert result.exit_code != 0
    assert "Repeat --target at least twice" in result.output


def test_nonstream_compare_help():
    """Test nonstream-compare subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["nonstream-compare", "--help"])
    assert result.exit_code == 0
    assert "--target" in result.output
    assert "--prompt" in result.output
    assert "--batch-size" in result.output


def test_vllm_compare_help():
    """Test vllm-compare command group help."""
    runner = CliRunner()
    result = runner.invoke(main, ["vllm-compare", "--help"])
    assert result.exit_code == 0
    assert "install-ascend" in result.output


def test_parity_gate_convert_core_telemetry_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["parity-gate", "convert-core-telemetry", "--help"])
    assert result.exit_code == 0
    assert "--input-json" in result.output
    assert "--label" in result.output
    assert "--hardware-family" in result.output


def test_vllm_compare_install_ascend_invokes_expected_steps(monkeypatch, tmp_path):
    """Install command should invoke benchmark extra install, pins, pip check, and smoke test."""
    calls: list[tuple[list[str], str | None]] = []

    def fake_run_checked_command(command: list[str], input_text: str | None = None) -> None:
        calls.append((command, input_text))

    monkeypatch.setattr("sagellm_benchmark.cli._run_checked_command", fake_run_checked_command)

    sagellm_root = tmp_path / "sagellm"
    wrapper_path = sagellm_root / "scripts" / "sagellm_with_ascend_env.sh"
    wrapper_path.parent.mkdir(parents=True)
    wrapper_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    wrapper_path.chmod(0o755)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "vllm-compare",
            "install-ascend",
            "--python-bin",
            sys.executable,
            "--sagellm-root",
            str(sagellm_root),
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 4
    assert calls[0][0][:5] == [sys.executable, "-m", "pip", "install", "-U"]
    assert "vllm-ascend-client" in calls[0][0][-1]
    assert calls[1][0][:5] == [sys.executable, "-m", "pip", "install", "-U"]
    assert "torch==2.7.1" in calls[1][0]
    assert calls[2][0] == [sys.executable, "-m", "pip", "check"]
    assert calls[3][0] == [str(wrapper_path), sys.executable, "-"]
    assert calls[3][1] is not None


def test_compare_safe_env_defaults_for_ascend(monkeypatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)

    _apply_compare_safe_env_defaults("ascend")

    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"


def test_compare_safe_env_defaults_preserve_user_overrides(monkeypatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://example.com/hf")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")

    _apply_compare_safe_env_defaults("ascend")

    assert os.environ["HF_ENDPOINT"] == "https://example.com/hf"
    assert os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "1"


def test_compare_safe_env_defaults_for_non_ascend(monkeypatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)

    _apply_compare_safe_env_defaults("cuda")

    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD" not in os.environ


def test_capture_target_runtime_artifacts_skips_invalid_core_telemetry(tmp_path: Path) -> None:
    info_payload = {
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
                "step_telemetry": {},
                "step_telemetry_entries": 0,
                "last_orchestration_step_id": 0,
            }
        }
    }

    def fake_fetch_json_probe(url: str, *, api_key: str, timeout_s: float):
        del url, api_key, timeout_s
        return info_payload

    from sagellm_benchmark import cli as cli_module

    original_fetch = cli_module._fetch_json_probe
    cli_module._fetch_json_probe = fake_fetch_json_probe
    try:
        runtime_artifacts = _capture_target_runtime_artifacts(
            label="sagellm",
            url="http://127.0.0.1:8901/v1",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            hardware_family="ascend",
            api_key="sagellm-benchmark",
            request_timeout=30.0,
            output_dir=tmp_path,
        )
    finally:
        cli_module._fetch_json_probe = original_fetch

    assert "info_json" in runtime_artifacts
    assert "core_telemetry_json" not in runtime_artifacts
    saved_info = json.loads(Path(runtime_artifacts["info_json"]).read_text(encoding="utf-8"))
    assert saved_info == info_payload


def test_validate_sagellm_explicit_decode_runtime_requires_enabled_feature_gate(
    tmp_path: Path,
) -> None:
    info_path = tmp_path / "sagellm_info.json"
    info_path.write_text(
        json.dumps(
            {
                "performance_mainline": {
                    "explicit_decode": {
                        "feature_gate": {
                            "feature_id": "runtime.native_decode.v1",
                            "default_enabled": False,
                            "enabled": False,
                            "kill_switch_active": False,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        _validate_sagellm_explicit_decode_runtime(
            label="sagellm",
            runtime_artifacts={"info_json": str(info_path)},
        )
    except click.ClickException as exc:
        assert "default_enabled=false" in str(exc)
    else:
        raise AssertionError("expected explicit decode validation to fail")


def test_validate_sagellm_explicit_decode_runtime_requires_step_telemetry(tmp_path: Path) -> None:
    info_path = tmp_path / "sagellm_info.json"
    telemetry_path = tmp_path / "sagellm_core_telemetry.json"
    info_path.write_text(
        json.dumps(
            {
                "performance_mainline": {
                    "explicit_decode": {
                        "feature_gate": {
                            "feature_id": "runtime.native_decode.v1",
                            "default_enabled": True,
                            "enabled": True,
                            "kill_switch_active": False,
                        }
                    },
                    "decode_runtime_diagnostics": {
                        "summary": {
                            "attention_batch_size": 1,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    telemetry_path.write_text(
        json.dumps(
            {
                "step_telemetry_entries": 0,
                "summary": {
                    "step_records": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        _validate_sagellm_explicit_decode_runtime(
            label="sagellm-ascend",
            runtime_artifacts={
                "info_json": str(info_path),
                "core_telemetry_json": str(telemetry_path),
            },
        )
    except click.ClickException as exc:
        assert "zero explicit decode step telemetry entries" in str(exc)
    else:
        raise AssertionError("expected explicit decode validation to fail")


def test_validate_sagellm_explicit_decode_runtime_accepts_valid_mainline(tmp_path: Path) -> None:
    info_path = tmp_path / "sagellm_info.json"
    telemetry_path = tmp_path / "sagellm_core_telemetry.json"
    info_path.write_text(
        json.dumps(
            {
                "performance_mainline": {
                    "explicit_decode": {
                        "feature_gate": {
                            "feature_id": "runtime.native_decode.v1",
                            "default_enabled": True,
                            "enabled": True,
                            "kill_switch_active": False,
                        }
                    },
                    "decode_runtime_diagnostics": {
                        "summary": {
                            "attention_batch_size": 1,
                            "attention_selected_implementation": "native-ascend",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    telemetry_path.write_text(
        json.dumps(
            {
                "step_telemetry_entries": 3,
                "summary": {
                    "step_records": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    _validate_sagellm_explicit_decode_runtime(
        label="sagellm",
        runtime_artifacts={
            "info_json": str(info_path),
            "core_telemetry_json": str(telemetry_path),
        },
    )


def test_upload_hf_help():
    """Test upload-hf subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["upload-hf", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "--input" in result.output
    assert "--token" in result.output


def test_publish_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["publish", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--dry-run" in result.output
    assert "--hf-dataset" in result.output
    assert "--website-dir" not in result.output


def test_run_mode_parameter():
    """Test that --mode parameter is available in run command."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--mode" in result.output
    assert "batch" in result.output
    assert "traffic" in result.output


def test_run_output_json_parameter():
    """Test that --output-json parameter is available in run command."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--output-json" in result.output


def test_mode_batch_validation():
    """Test that batch mode is a valid choice."""
    runner = CliRunner()
    # Note: This will fail without a proper engine, but we just want to verify the parameter is accepted
    # The actual validation happens during execution
    result = runner.invoke(main, ["run", "--mode", "batch", "--help"])
    assert result.exit_code == 0


def test_mode_traffic_validation():
    """Test that traffic mode is a valid choice."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--mode", "traffic", "--help"])
    assert result.exit_code == 0


def test_run_generates_canonical_artifacts_without_leaderboard_exports(monkeypatch):
    class FakeEngine:
        def __init__(self, config):
            self.config = config
            self.is_running = False

        async def start(self):
            self.is_running = True

    class FakeEngineConfig:
        def __init__(self, **kwargs):
            self.model_path = kwargs["model_path"]

    class FakeBenchmarkRunner:
        def __init__(self, config):
            self.config = config

        async def run(self):
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            metrics = AggregatedMetrics(
                avg_ttft_ms=9.0,
                avg_tbt_ms=2.0,
                avg_throughput_tps=80.0,
                output_throughput_tps=80.0,
                total_requests=2,
                successful_requests=2,
                failed_requests=0,
            )
            metrics_path = self.config.output_dir / "vllm_random_b1_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "avg_ttft_ms": metrics.avg_ttft_ms,
                        "avg_tbt_ms": metrics.avg_tbt_ms,
                        "avg_throughput_tps": metrics.avg_throughput_tps,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            summary_path = self.config.output_dir / "benchmark_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "workloads": {"vllm_random_b1": {"avg_ttft_ms": metrics.avg_ttft_ms}},
                        "overall": {"total_workloads": 1, "total_requests": 2},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {"vllm_random_b1": metrics}

    monkeypatch.setitem(
        sys.modules,
        "sagellm_core",
        SimpleNamespace(LLMEngine=FakeEngine, LLMEngineConfig=FakeEngineConfig),
    )
    monkeypatch.setattr("sagellm_benchmark.runner.BenchmarkRunner", FakeBenchmarkRunner)

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("run_out")
        result = runner.invoke(
            main,
            [
                "run",
                "--profile",
                "vllm_random",
                "--batch-size",
                "1",
                "--backend",
                "cpu",
                "--model",
                "sshleifer/tiny-gpt2",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "vllm_random_b1.canonical.json").exists()
        assert not (output_dir / "vllm_random_b1_leaderboard.json").exists()
        assert not (output_dir / "leaderboard_manifest.json").exists()
        payload = json.loads(
            (output_dir / "vllm_random_b1.canonical.json").read_text(encoding="utf-8")
        )
        assert payload["schema_version"] == "canonical-benchmark-result/v2"
        assert payload["producer"]["command"] == "run"
        assert "leaderboard_json" not in payload["artifacts"]


@pytest.mark.parametrize(
    ("argv", "expected_message"),
    [
        (
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
                "--profile",
                "vllm_custom",
            ],
            "vllm_custom requires --dataset-path",
        ),
        (
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
                "--profile",
                "vllm_custom",
                "--dataset-path",
                "./sharegpt.json",
            ],
            "vllm_custom requires --num-prompts",
        ),
    ],
)
def test_profile_custom_requires_explicit_shape(
    argv: list[str],
    expected_message: str,
) -> None:
    runner = CliRunner()

    result = runner.invoke(main, argv)

    assert result.exit_code != 0
    assert expected_message in result.output


def test_run_publish_dry_run_rejects_non_compare_leaderboard_export(monkeypatch):
    class FakeEngine:
        def __init__(self, config):
            self.config = config

        async def start(self):
            return None

    class FakeEngineConfig:
        def __init__(self, **kwargs):
            self.model_path = kwargs["model_path"]

    class FakeBenchmarkRunner:
        def __init__(self, config):
            self.config = config

        async def run(self):
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            metrics = AggregatedMetrics(
                avg_ttft_ms=9.0,
                avg_tbt_ms=2.0,
                avg_throughput_tps=80.0,
                output_throughput_tps=80.0,
                total_requests=2,
                successful_requests=2,
                failed_requests=0,
            )
            (self.config.output_dir / "vllm_random_b1_metrics.json").write_text(
                json.dumps({"avg_ttft_ms": metrics.avg_ttft_ms}, indent=2),
                encoding="utf-8",
            )
            (self.config.output_dir / "benchmark_summary.json").write_text(
                json.dumps(
                    {
                        "workloads": {"vllm_random_b1": {"avg_ttft_ms": metrics.avg_ttft_ms}},
                        "overall": {"total_workloads": 1, "total_requests": 2},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {"vllm_random_b1": metrics}

    monkeypatch.setitem(
        sys.modules,
        "sagellm_core",
        SimpleNamespace(LLMEngine=FakeEngine, LLMEngineConfig=FakeEngineConfig),
    )
    monkeypatch.setattr("sagellm_benchmark.runner.BenchmarkRunner", FakeBenchmarkRunner)

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("run_publish_out")

        result = runner.invoke(
            main,
            [
                "run",
                "--profile",
                "vllm_random",
                "--batch-size",
                "1",
                "--backend",
                "cpu",
                "--model",
                "sshleifer/tiny-gpt2",
                "--output",
                str(output_dir),
                "--publish",
                "--publish-dry-run",
            ],
        )

        assert result.exit_code != 0
        assert "No leaderboard_manifest.json found under" in result.output
        assert not (output_dir / "publish" / "website-ready" / "leaderboard_single.json").exists()


def test_upload_hf_dry_run_requires_standard_manifest_exports() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("exports")
        output_dir.mkdir()
        result = runner.invoke(main, ["upload-hf", "--input", str(output_dir), "--dry-run"])

        assert result.exit_code != 0
        assert "leaderboard_manifest.json" in result.output


def test_upload_hf_is_publish_compatibility_wrapper(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_publish_workflow(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("sagellm_benchmark.cli._run_publish_workflow", fake_run_publish_workflow)

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_dir = Path("exports")
        output_dir.mkdir()

        result = runner.invoke(
            main,
            [
                "upload-hf",
                "--input",
                str(output_dir),
                "--dataset",
                "demo/repo",
                "--token",
                "secret",
                "--private",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0
    assert "Compatibility layer" in result.output
    assert captured == {
        "benchmark_output_dir": Path("exports"),
        "publish_hf_dataset": "demo/repo",
        "publish_hf_token": "secret",
        "publish_hf_private": True,
        "publish_dry_run": True,
    }
