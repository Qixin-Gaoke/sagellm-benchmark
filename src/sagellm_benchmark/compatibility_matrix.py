"""Compatibility matrix gate for protocol/version/error/stream regression checks."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any


def _parse_bool_env(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value for {key}: {value!r}. Use one of: 1/0, true/false, yes/no, on/off"
    )


@dataclass(frozen=True)
class CompatibilityMatrixConfig:
    enabled: bool
    kill_switch_active: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> CompatibilityMatrixConfig:
        source = env or dict(os.environ)
        enabled_raw = source.get("SAGELLM_BENCH_COMPAT", "0")
        kill_raw = source.get("SAGELLM_BENCH_COMPAT_KILL", "0")
        return cls(
            enabled=_parse_bool_env(enabled_raw, "SAGELLM_BENCH_COMPAT"),
            kill_switch_active=_parse_bool_env(kill_raw, "SAGELLM_BENCH_COMPAT_KILL"),
        )


@dataclass(frozen=True)
class CompatibilityCase:
    name: str
    protocol_version: str
    negotiated_version: str | None
    endpoint_type: str
    stream: bool
    expected_success: bool
    observed_success: bool
    expected_error_code: str | None = None
    observed_error_code: str | None = None
    response_text: str = ""
    consistency_group: str | None = None


@dataclass(frozen=True)
class CompatibilityCheck:
    name: str
    passed: bool
    detail: str


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def evaluate_compatibility_matrix(
    cases: list[CompatibilityCase],
) -> dict[str, Any]:
    if not cases:
        raise ValueError("Compatibility matrix requires at least one case.")

    checks: list[CompatibilityCheck] = []
    error_counter: Counter[str] = Counter()
    version_mismatch_total = 0
    passed_cases = 0

    grouped: dict[str, list[CompatibilityCase]] = {}
    for case in cases:
        if case.observed_error_code:
            error_counter[case.observed_error_code] += 1
        elif not case.observed_success:
            error_counter["unknown"] += 1

        if case.negotiated_version is not None and case.negotiated_version != case.protocol_version:
            version_mismatch_total += 1

        success_match = case.expected_success == case.observed_success
        error_match = case.expected_error_code == case.observed_error_code
        case_passed = success_match and error_match
        if case_passed:
            passed_cases += 1

        checks.append(
            CompatibilityCheck(
                name=case.name,
                passed=case_passed,
                detail=(
                    f"expected_success={case.expected_success} observed_success={case.observed_success}; "
                    f"expected_error_code={case.expected_error_code!r} observed_error_code={case.observed_error_code!r}; "
                    f"protocol_version={case.protocol_version!r} negotiated_version={case.negotiated_version!r}"
                ),
            )
        )
        if case.consistency_group:
            grouped.setdefault(case.consistency_group, []).append(case)

    stream_consistency_checks: list[dict[str, Any]] = []
    for group_name, grouped_cases in grouped.items():
        if len(grouped_cases) < 2:
            continue
        normalized_outputs = {_normalize_text(case.response_text) for case in grouped_cases}
        success_profiles = {
            (case.stream, case.observed_success, case.observed_error_code) for case in grouped_cases
        }
        consistent = len(normalized_outputs) <= 1 and len(success_profiles) == len(grouped_cases)
        if all(case.observed_success for case in grouped_cases):
            consistent = len(normalized_outputs) == 1
        stream_consistency_checks.append(
            {
                "group": group_name,
                "passed": consistent,
                "cases": [case.name for case in grouped_cases],
            }
        )

    pass_rate = passed_cases / len(cases)
    overall_passed = all(check.passed for check in checks) and all(
        item["passed"] for item in stream_consistency_checks
    )

    return {
        "overall_passed": overall_passed,
        "compatibility_pass_rate": pass_rate,
        "version_mismatch_total": version_mismatch_total,
        "error_code_distribution": dict(sorted(error_counter.items())),
        "checks": [check.__dict__ for check in checks],
        "stream_nonstream_consistency": stream_consistency_checks,
    }
