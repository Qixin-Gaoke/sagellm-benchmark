"""Tests for benchmark parity-gate schema and evaluation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sagellm_benchmark.parity_gate import (
    GateFailureCategory,
    ParityRunArtifact,
    ParityScenarioMetrics,
    build_default_cuda_decode_gate,
    build_parity_run_artifact_from_e2e_payload,
    evaluate_parity_gate,
    load_parity_run_artifact,
)


def _artifact(
    *,
    label: str,
    tbt: float,
    output_throughput: float,
    has_step_evidence: bool,
    fallback_rate: float | None = 0.0,
    has_fallback_evidence: bool = True,
    correctness_pass_rate: float = 1.0,
) -> ParityRunArtifact:
    scenarios = []
    for batch_size, suffix in ((1, "vllm_random_b1"), (2, "vllm_random_b2"), (4, "vllm_random_b4")):
        scenarios.append(
            ParityScenarioMetrics(
                scenario_name=suffix,
                batch_size=batch_size,
                avg_tbt_ms=tbt,
                output_throughput_tps=output_throughput,
                correctness_pass_rate=correctness_pass_rate,
                fallback_rate=fallback_rate,
                has_step_evidence=has_step_evidence,
                has_fallback_evidence=has_fallback_evidence,
            )
        )
        scenarios.append(
            ParityScenarioMetrics(
                scenario_name=f"vllm_sharegpt_b{batch_size}",
                batch_size=batch_size,
                avg_tbt_ms=tbt,
                output_throughput_tps=output_throughput,
                correctness_pass_rate=correctness_pass_rate,
                fallback_rate=fallback_rate,
                has_step_evidence=has_step_evidence,
                has_fallback_evidence=has_fallback_evidence,
            )
        )
    return ParityRunArtifact(
        label=label,
        engine_family=label,
        hardware_family="cuda",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        scenarios=scenarios,
    )


def test_default_cuda_gate_covers_bs_1_2_4() -> None:
    gate = build_default_cuda_decode_gate()

    assert gate.hardware_family == "cuda"
    assert gate.warmup_rounds == 3
    assert gate.measured_rounds == 10
    assert {scenario.batch_size for scenario in gate.scenarios} == {1, 2, 4}


def test_gate_fails_without_step_evidence() -> None:
    gate = build_default_cuda_decode_gate()
    candidate = _artifact(
        label="sagellm", tbt=10.0, output_throughput=100.0, has_step_evidence=False
    )
    reference = _artifact(label="vllm", tbt=10.0, output_throughput=100.0, has_step_evidence=True)

    result = evaluate_parity_gate(gate, candidate, [reference])

    assert result.passed is False
    assert all(item.category == GateFailureCategory.TELEMETRY for item in result.results)


def test_gate_reports_telemetry_and_performance_regression_together() -> None:
    gate = build_default_cuda_decode_gate()
    candidate = _artifact(
        label="sagellm",
        tbt=12.0,
        output_throughput=80.0,
        has_step_evidence=False,
    )
    reference = _artifact(label="vllm", tbt=10.0, output_throughput=100.0, has_step_evidence=True)

    result = evaluate_parity_gate(gate, candidate, [reference])

    assert result.passed is False
    by_scenario: dict[str, set[GateFailureCategory]] = {}
    for item in result.results:
        by_scenario.setdefault(item.scenario_name, set()).add(item.category)
    assert by_scenario
    assert all(
        categories == {GateFailureCategory.TELEMETRY, GateFailureCategory.PERFORMANCE}
        for categories in by_scenario.values()
    )


def test_gate_fails_on_performance_regression() -> None:
    gate = build_default_cuda_decode_gate()
    candidate = _artifact(label="sagellm", tbt=12.0, output_throughput=80.0, has_step_evidence=True)
    reference = _artifact(label="vllm", tbt=10.0, output_throughput=100.0, has_step_evidence=True)

    result = evaluate_parity_gate(gate, candidate, [reference])

    assert result.passed is False
    assert all(item.category == GateFailureCategory.PERFORMANCE for item in result.results)


def test_gate_passes_when_within_defined_band() -> None:
    gate = build_default_cuda_decode_gate()
    candidate = _artifact(label="sagellm", tbt=10.3, output_throughput=98.0, has_step_evidence=True)
    references = [
        _artifact(label="vllm", tbt=10.0, output_throughput=100.0, has_step_evidence=True),
        _artifact(label="sglang", tbt=10.5, output_throughput=98.0, has_step_evidence=True),
    ]

    result = evaluate_parity_gate(gate, candidate, references)

    assert result.passed is True
    assert all(item.category == GateFailureCategory.PASS for item in result.results)


def test_gate_distinguishes_fallback_failure() -> None:
    gate = build_default_cuda_decode_gate()
    candidate = _artifact(
        label="sagellm",
        tbt=10.0,
        output_throughput=100.0,
        has_step_evidence=True,
        fallback_rate=0.25,
    )
    reference = _artifact(label="vllm", tbt=10.0, output_throughput=100.0, has_step_evidence=True)

    result = evaluate_parity_gate(gate, candidate, [reference])

    assert result.passed is False
    assert all(item.category == GateFailureCategory.FALLBACK for item in result.results)


def test_gate_requires_fallback_evidence_instead_of_assuming_zero() -> None:
    gate = build_default_cuda_decode_gate()
    candidate = _artifact(
        label="sagellm",
        tbt=10.0,
        output_throughput=100.0,
        has_step_evidence=True,
        fallback_rate=None,
        has_fallback_evidence=False,
    )
    reference = _artifact(label="vllm", tbt=10.0, output_throughput=100.0, has_step_evidence=True)

    result = evaluate_parity_gate(gate, candidate, [reference])

    assert result.passed is False
    assert all(item.category == GateFailureCategory.FALLBACK for item in result.results)
    assert all("fallback evidence is missing" in item.message for item in result.results)


def test_legacy_e2e_conversion_keeps_throughput_undistorted(tmp_path) -> None:
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "kind": "e2e",
                "metadata": {
                    "label": "sagellm",
                    "hardware": "cuda",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                },
                "rows": [
                    {
                        "scenario": "short",
                        "batch_size": 4,
                        "tbt_ms": 10.0,
                        "throughput_tps": 111.0,
                        "correctness_pass_rate": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = load_parity_run_artifact(candidate_path)

    assert artifact.scenarios[0].scenario_name == "vllm_random_b4"
    assert artifact.scenarios[0].output_throughput_tps == 111.0
    assert artifact.scenarios[0].fallback_rate is None
    assert artifact.scenarios[0].has_fallback_evidence is False


def test_legacy_e2e_conversion_prefers_output_throughput_when_present(tmp_path) -> None:
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "kind": "e2e",
                "metadata": {
                    "label": "sagellm",
                    "hardware": "cuda",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                },
                "rows": [
                    {
                        "scenario": "short",
                        "batch_size": 4,
                        "tbt_ms": 10.0,
                        "throughput_tps": 111.0,
                        "output_throughput_tps": 222.0,
                        "correctness_pass_rate": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = load_parity_run_artifact(candidate_path)

    assert artifact.scenarios[0].output_throughput_tps == 222.0


def test_legacy_e2e_artifact_can_hit_gate_scenarios(tmp_path) -> None:
    gate = build_default_cuda_decode_gate()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "kind": "e2e",
                "metadata": {
                    "label": "sagellm",
                    "hardware": "cuda",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                },
                "rows": [
                    {
                        "scenario": scenario,
                        "batch_size": batch_size,
                        "tbt_ms": 12.0,
                        "throughput_tps": 80.0,
                        "correctness_pass_rate": 1.0,
                    }
                    for scenario in ("short", "long")
                    for batch_size in (1, 2, 4)
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate = load_parity_run_artifact(candidate_path)
    reference = _artifact(label="vllm", tbt=10.0, output_throughput=100.0, has_step_evidence=True)

    result = evaluate_parity_gate(gate, candidate, [reference])

    assert result.passed is False
    assert all(item.category != GateFailureCategory.CAPABILITY for item in result.results)


def test_build_parity_run_artifact_from_e2e_payload_succeeds() -> None:
    artifact = build_parity_run_artifact_from_e2e_payload(
        {
            "kind": "e2e",
            "label": "sagellm",
            "models": ["Qwen/Qwen2.5-0.5B-Instruct"],
            "mode": "live-compare",
            "url": "http://127.0.0.1:8901/v1",
            "rows": [
                {
                    "scenario": "vllm_random_b1",
                    "batch_size": 1,
                    "tbt_ms": 2.0,
                    "output_throughput_tps": 100.0,
                    "successful_requests": 1,
                    "failed_requests": 0,
                }
            ],
        },
        hardware_family="cuda",
    )

    assert artifact.schema_version == "parity-run/v1"
    assert artifact.hardware_family == "cuda"
    assert artifact.scenarios[0].scenario_name == "vllm_random_b1"
    assert artifact.scenarios[0].has_step_evidence is False


def test_build_parity_run_artifact_from_e2e_payload_fails_on_missing_field() -> None:
    with pytest.raises(ValueError, match="requires output_throughput_tps or throughput_tps"):
        build_parity_run_artifact_from_e2e_payload(
            {
                "kind": "e2e",
                "label": "sagellm",
                "models": ["Qwen/Qwen2.5-0.5B-Instruct"],
                "rows": [
                    {
                        "scenario": "vllm_random_b1",
                        "batch_size": 1,
                        "tbt_ms": 2.0,
                        "successful_requests": 1,
                        "failed_requests": 0,
                    }
                ],
            },
            hardware_family="cuda",
        )


def test_load_parity_run_artifact_rejects_schema_mismatch(tmp_path) -> None:
    payload_path = tmp_path / "candidate.parity.json"
    payload_path.write_text(
        json.dumps(
            {
                "schema_version": "parity-run/v1",
                "label": "sagellm",
                "engine_family": "sagellm",
                "hardware_family": "cuda",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "scenarios": [
                    {
                        "scenario_name": "vllm_random_b1",
                        "batch_size": 1,
                        "avg_tbt_ms": 2.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_parity_run_artifact(payload_path)
