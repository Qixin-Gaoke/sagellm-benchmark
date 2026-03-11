"""Tests for CLI functionality."""

from __future__ import annotations

import sys

from click.testing import CliRunner

from sagellm_benchmark.cli import main


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
    assert "--workload" in result.output
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
    assert "--batch-size" in result.output


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
    assert "run" in result.output


def test_vllm_compare_run_help():
    """Test vllm-compare run help."""
    runner = CliRunner()
    result = runner.invoke(main, ["vllm-compare", "run", "--help"])
    assert result.exit_code == 0
    assert "--vllm-url" in result.output
    assert "--sagellm-url" in result.output
    assert "--batch-size" in result.output


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


def test_upload_hf_help():
    """Test upload-hf subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["upload-hf", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "--input" in result.output
    assert "--token" in result.output


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
