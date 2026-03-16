"""Reusable parity-gate schema and evaluation helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _CompatModel(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=False)


class GateFailureCategory(StrEnum):
    PASS = "pass"
    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    FALLBACK = "fallback"
    CAPABILITY = "capability"
    TELEMETRY = "telemetry"
    NO_REFERENCE = "no_reference"


class DecodeParityScenario(_CompatModel):
    name: str = Field(description="Stable scenario identifier.")
    prompt_tokens: int = Field(ge=1, description="Prompt length in tokens.")
    output_tokens: int = Field(ge=1, description="Output length in tokens.")
    batch_size: int = Field(ge=1, description="Decode batch size.")


class DecodeParityThresholds(_CompatModel):
    max_tbt_ratio_vs_best_reference: float = Field(default=1.05, ge=0.0)
    min_output_throughput_ratio_vs_best_reference: float = Field(default=0.95, ge=0.0)
    min_correctness_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    max_fallback_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    require_step_evidence: bool = Field(default=True)


class DecodeParityGate(_CompatModel):
    gate_id: str
    hardware_family: str
    workload_id: str
    warmup_rounds: int = Field(ge=0)
    measured_rounds: int = Field(ge=1)
    comparison_mode: str = Field(default="best_reference")
    scenarios: list[DecodeParityScenario]
    thresholds: DecodeParityThresholds
    required_artifacts: tuple[str, ...] = Field(
        default=("summary", "step_telemetry", "fallback_report"),
    )
    exit_rule: str = Field(
        default=(
            "A candidate passes only if every scenario satisfies correctness, fallback, "
            "step-evidence, and performance thresholds against the best reference."
        )
    )


class ParityScenarioMetrics(_CompatModel):
    scenario_name: str
    batch_size: int = Field(ge=1)
    avg_tbt_ms: float = Field(ge=0.0)
    output_throughput_tps: float = Field(ge=0.0)
    correctness_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    fallback_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    has_step_evidence: bool = Field(default=False)
    has_fallback_evidence: bool = Field(default=False)
    capability_gaps: list[str] = Field(default_factory=list)


class ParityRunArtifact(_CompatModel):
    schema_version: str = Field(default="parity-run/v1")
    label: str
    engine_family: str
    hardware_family: str
    model: str
    scenarios: list[ParityScenarioMetrics]
    metadata: dict[str, Any] | None = Field(default=None)


class ScenarioGateResult(_CompatModel):
    scenario_name: str
    batch_size: int
    category: GateFailureCategory
    passed: bool
    message: str
    candidate_tbt_ms: float | None = None
    best_reference_tbt_ms: float | None = None
    candidate_output_throughput_tps: float | None = None
    best_reference_output_throughput_tps: float | None = None


class ParityGateEvaluation(_CompatModel):
    gate_id: str
    candidate_label: str
    passed: bool
    results: list[ScenarioGateResult]


def build_parity_run_artifact_from_e2e_payload(
    payload: dict[str, Any],
    *,
    hardware_family: str,
    engine_family: str | None = None,
    has_step_evidence: bool = False,
    has_fallback_evidence: bool = False,
) -> ParityRunArtifact:
    """Build a strict parity-run/v1 artifact from a compare e2e payload."""
    if payload.get("kind") != "e2e":
        raise ValueError("Parity export requires an e2e compare payload")

    normalized_hardware_family = hardware_family.strip()
    if not normalized_hardware_family:
        raise ValueError("Parity export requires a non-empty hardware_family")

    label = str(payload.get("label") or "").strip()
    if not label:
        raise ValueError("Parity export requires payload.label")

    models = payload.get("models")
    if not isinstance(models, list) or not models or not str(models[0]).strip():
        raise ValueError("Parity export requires payload.models[0]")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Parity export requires a non-empty payload.rows list")

    scenarios = [
        _build_parity_scenario_metrics_from_e2e_row(
            row,
            has_step_evidence=has_step_evidence,
            has_fallback_evidence=has_fallback_evidence,
        )
        for row in rows
    ]
    scenario_groups = {
        "mainline": sum(
            1
            for row in rows
            if isinstance(row, dict) and str(row.get("scenario_source")) == "mainline"
        ),
        "supplement": sum(
            1
            for row in rows
            if isinstance(row, dict) and str(row.get("scenario_source")) == "supplement"
        ),
    }

    return ParityRunArtifact(
        label=label,
        engine_family=(engine_family or label).strip(),
        hardware_family=normalized_hardware_family,
        model=str(models[0]).strip(),
        scenarios=scenarios,
        metadata={
            "source_kind": payload.get("kind"),
            "source_mode": payload.get("mode"),
            "source_url": payload.get("url"),
            "scenario_groups": scenario_groups,
            "generated_by": "sagellm-benchmark live compare",
        },
    )


def build_default_cuda_decode_gate() -> DecodeParityGate:
    scenarios: list[DecodeParityScenario] = []
    for batch_size in (1, 2, 4):
        scenarios.append(
            DecodeParityScenario(
                name=f"vllm_random_b{batch_size}",
                prompt_tokens=128,
                output_tokens=128,
                batch_size=batch_size,
            )
        )
        scenarios.append(
            DecodeParityScenario(
                name=f"vllm_sharegpt_b{batch_size}",
                prompt_tokens=256,
                output_tokens=128,
                batch_size=batch_size,
            )
        )

    return DecodeParityGate(
        gate_id="cuda_decode_parity_v1",
        hardware_family="cuda",
        workload_id="decode_bs1_bs2_bs4",
        warmup_rounds=3,
        measured_rounds=10,
        scenarios=scenarios,
        thresholds=DecodeParityThresholds(),
    )


def load_parity_run_artifact(path: str | Path) -> ParityRunArtifact:
    payload_path = Path(path)
    with payload_path.open() as handle:
        payload = json.load(handle)

    if payload.get("schema_version") == "parity-run/v1":
        return ParityRunArtifact.model_validate(payload)
    if payload.get("kind") == "e2e":
        return _convert_e2e_payload(payload)
    raise ValueError(f"Unsupported parity artifact: {payload_path}")


def evaluate_parity_gate(
    gate: DecodeParityGate,
    candidate: ParityRunArtifact,
    references: list[ParityRunArtifact],
) -> ParityGateEvaluation:
    reference_index: dict[tuple[str, int], list[ParityScenarioMetrics]] = defaultdict(list)
    for reference in references:
        for scenario in reference.scenarios:
            reference_index[(scenario.scenario_name, scenario.batch_size)].append(scenario)

    candidate_index = {
        (scenario.scenario_name, scenario.batch_size): scenario for scenario in candidate.scenarios
    }

    results: list[ScenarioGateResult] = []
    for scenario_def in gate.scenarios:
        key = (scenario_def.name, scenario_def.batch_size)
        candidate_metrics = candidate_index.get(key)
        if candidate_metrics is None:
            results.append(
                ScenarioGateResult(
                    scenario_name=scenario_def.name,
                    batch_size=scenario_def.batch_size,
                    category=GateFailureCategory.CAPABILITY,
                    passed=False,
                    message="candidate artifact is missing this scenario",
                )
            )
            continue

        # Keep evidence categories independent so legacy artifacts can report
        # telemetry/fallback gaps without hiding a real performance regression.
        scenario_findings: list[ScenarioGateResult] = []
        if candidate_metrics.capability_gaps:
            scenario_findings.append(
                ScenarioGateResult(
                    scenario_name=scenario_def.name,
                    batch_size=scenario_def.batch_size,
                    category=GateFailureCategory.CAPABILITY,
                    passed=False,
                    message=(
                        "candidate reports capability gaps: "
                        + ", ".join(candidate_metrics.capability_gaps)
                    ),
                )
            )

        if candidate_metrics.correctness_pass_rate < gate.thresholds.min_correctness_pass_rate:
            scenario_findings.append(
                ScenarioGateResult(
                    scenario_name=scenario_def.name,
                    batch_size=scenario_def.batch_size,
                    category=GateFailureCategory.CORRECTNESS,
                    passed=False,
                    message=(
                        "correctness pass rate below threshold: "
                        f"{candidate_metrics.correctness_pass_rate:.2f} < "
                        f"{gate.thresholds.min_correctness_pass_rate:.2f}"
                    ),
                )
            )

        if not candidate_metrics.has_fallback_evidence:
            scenario_findings.append(
                ScenarioGateResult(
                    scenario_name=scenario_def.name,
                    batch_size=scenario_def.batch_size,
                    category=GateFailureCategory.FALLBACK,
                    passed=False,
                    message="fallback evidence is missing; legacy artifacts cannot prove fallback_rate",
                )
            )
        elif (
            candidate_metrics.fallback_rate is not None
            and candidate_metrics.fallback_rate > gate.thresholds.max_fallback_rate
        ):
            scenario_findings.append(
                ScenarioGateResult(
                    scenario_name=scenario_def.name,
                    batch_size=scenario_def.batch_size,
                    category=GateFailureCategory.FALLBACK,
                    passed=False,
                    message=(
                        "fallback rate above threshold: "
                        f"{candidate_metrics.fallback_rate:.2f} > "
                        f"{gate.thresholds.max_fallback_rate:.2f}"
                    ),
                )
            )

        if gate.thresholds.require_step_evidence and not candidate_metrics.has_step_evidence:
            scenario_findings.append(
                ScenarioGateResult(
                    scenario_name=scenario_def.name,
                    batch_size=scenario_def.batch_size,
                    category=GateFailureCategory.TELEMETRY,
                    passed=False,
                    message="step-level evidence is required but missing",
                )
            )

        reference_metrics = reference_index.get(key, [])
        if not reference_metrics:
            scenario_findings.append(
                ScenarioGateResult(
                    scenario_name=scenario_def.name,
                    batch_size=scenario_def.batch_size,
                    category=GateFailureCategory.NO_REFERENCE,
                    passed=False,
                    message="no reference artifact provides this scenario",
                )
            )
        else:
            best_reference_tbt = min(item.avg_tbt_ms for item in reference_metrics)
            best_reference_output = max(item.output_throughput_tps for item in reference_metrics)
            candidate_tbt_ratio = (
                candidate_metrics.avg_tbt_ms / best_reference_tbt
                if best_reference_tbt > 0
                else float("inf")
            )
            candidate_output_ratio = (
                candidate_metrics.output_throughput_tps / best_reference_output
                if best_reference_output > 0
                else 0.0
            )

            performance_passed = (
                candidate_tbt_ratio <= gate.thresholds.max_tbt_ratio_vs_best_reference
                and candidate_output_ratio
                >= gate.thresholds.min_output_throughput_ratio_vs_best_reference
            )
            if not performance_passed:
                scenario_findings.append(
                    ScenarioGateResult(
                        scenario_name=scenario_def.name,
                        batch_size=scenario_def.batch_size,
                        category=GateFailureCategory.PERFORMANCE,
                        passed=False,
                        message=(
                            "performance outside gate: "
                            f"tbt_ratio={candidate_tbt_ratio:.3f}, "
                            f"output_ratio={candidate_output_ratio:.3f}"
                        ),
                        candidate_tbt_ms=candidate_metrics.avg_tbt_ms,
                        best_reference_tbt_ms=best_reference_tbt,
                        candidate_output_throughput_tps=candidate_metrics.output_throughput_tps,
                        best_reference_output_throughput_tps=best_reference_output,
                    )
                )
            elif not scenario_findings:
                scenario_findings.append(
                    ScenarioGateResult(
                        scenario_name=scenario_def.name,
                        batch_size=scenario_def.batch_size,
                        category=GateFailureCategory.PASS,
                        passed=True,
                        message="performance within gate",
                        candidate_tbt_ms=candidate_metrics.avg_tbt_ms,
                        best_reference_tbt_ms=best_reference_tbt,
                        candidate_output_throughput_tps=candidate_metrics.output_throughput_tps,
                        best_reference_output_throughput_tps=best_reference_output,
                    )
                )

        if scenario_findings:
            if any(not finding.passed for finding in scenario_findings):
                scenario_findings = [
                    finding
                    for finding in scenario_findings
                    if finding.category != GateFailureCategory.PASS
                ]
            results.extend(scenario_findings)
            continue

        results.append(
            ScenarioGateResult(
                scenario_name=scenario_def.name,
                batch_size=scenario_def.batch_size,
                category=GateFailureCategory.PASS,
                passed=True,
                message="scenario passed all gate checks",
            )
        )

    return ParityGateEvaluation(
        gate_id=gate.gate_id,
        candidate_label=candidate.label,
        passed=all(result.passed for result in results),
        results=results,
    )


def _convert_e2e_payload(payload: dict[str, Any]) -> ParityRunArtifact:
    rows = payload.get("rows") or []
    scenarios: list[ParityScenarioMetrics] = []
    for row in rows:
        scenarios.append(
            _build_parity_scenario_metrics_from_e2e_row(
                row,
                has_step_evidence=False,
                has_fallback_evidence=False,
                strict=False,
            )
        )

    models = payload.get("models") or [payload.get("model") or "unknown"]
    return ParityRunArtifact(
        label=str(payload.get("label") or "candidate"),
        engine_family=str(payload.get("label") or "unknown"),
        hardware_family=str(payload.get("hardware_family") or "unknown"),
        model=str(models[0]),
        scenarios=scenarios,
        metadata={
            "source_kind": payload.get("kind"),
            "source_mode": payload.get("mode"),
            "note": "Converted from compare-record/compare e2e payload. Step evidence remains missing until runtime artifacts are attached.",
        },
    )


def _build_parity_scenario_metrics_from_e2e_row(
    row: dict[str, Any],
    *,
    has_step_evidence: bool,
    has_fallback_evidence: bool,
    strict: bool = True,
) -> ParityScenarioMetrics:
    if not isinstance(row, dict):
        raise ValueError("Parity export requires each e2e row to be a JSON object")

    required_fields = ("scenario", "batch_size", "tbt_ms")
    missing_fields = [field_name for field_name in required_fields if field_name not in row]
    if missing_fields:
        raise ValueError(f"Parity export row missing required fields: {missing_fields}")

    batch_size = int(row["batch_size"])
    scenario = str(row["scenario"])
    if scenario == "short":
        scenario = "vllm_random"
    elif scenario == "long":
        scenario = "vllm_sharegpt"
    scenario_name = scenario if "_b" in scenario else f"{scenario}_b{batch_size}"

    output_throughput = row.get("output_throughput_tps")
    if output_throughput is None:
        output_throughput = row.get("throughput_tps")
    if output_throughput is None and strict:
        raise ValueError(
            "Parity export row requires output_throughput_tps or throughput_tps: "
            f"scenario={scenario_name}"
        )

    correctness_pass_rate = row.get("correctness_pass_rate")
    if correctness_pass_rate is None:
        has_success_fields = "successful_requests" in row or "failed_requests" in row
        if strict and not has_success_fields:
            raise ValueError(
                "Parity export row requires correctness_pass_rate or successful_requests/failed_requests: "
                f"scenario={scenario_name}"
            )
        total_requests = int(row.get("successful_requests") or 0) + int(
            row.get("failed_requests") or 0
        )
        correctness_pass_rate = (
            float(row.get("successful_requests") or 0) / total_requests if total_requests else 1.0
        )

    return ParityScenarioMetrics(
        scenario_name=scenario_name,
        batch_size=batch_size,
        avg_tbt_ms=float(row.get("tbt_ms") or 0.0),
        output_throughput_tps=float(output_throughput or 0.0),
        correctness_pass_rate=float(correctness_pass_rate),
        fallback_rate=None,
        has_step_evidence=has_step_evidence,
        has_fallback_evidence=has_fallback_evidence,
        capability_gaps=list(row.get("capability_gaps") or []),
    )
