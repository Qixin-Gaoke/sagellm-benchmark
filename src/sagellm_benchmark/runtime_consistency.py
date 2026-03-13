"""Minimal live endpoint consistency checks for decode runtime evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must decode to a JSON object: {path}")
    return payload


def extract_runtime_info_payload(info_payload: dict[str, Any]) -> dict[str, Any]:
    """Return the engine-level runtime payload from either engine or gateway /info."""
    if isinstance(info_payload.get("performance_mainline"), dict):
        return info_payload

    registered_engines = info_payload.get("registered_engines")
    if not isinstance(registered_engines, list):
        raise ValueError("info_json is missing performance_mainline and registered_engines")

    engine_payloads = []
    for entry in registered_engines:
        if not isinstance(entry, dict):
            continue
        nested_info = entry.get("info")
        if isinstance(nested_info, dict) and isinstance(
            nested_info.get("performance_mainline"), dict
        ):
            engine_payloads.append(nested_info)

    if len(engine_payloads) != 1:
        raise ValueError(
            "gateway /info must expose exactly one engine info payload with performance_mainline; "
            f"found {len(engine_payloads)}"
        )
    return engine_payloads[0]


def _get_required_dict(
    payload: dict[str, Any], path: tuple[str, ...], *, label: str
) -> dict[str, Any]:
    current: Any = payload
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"{label} missing required object: {'.'.join(traversed)}")
        current = current[key]
    if not isinstance(current, dict):
        raise ValueError(f"{label} field is not an object: {'.'.join(path)}")
    return current


def _as_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc


def _as_bool(value: Any) -> bool:
    return bool(value)


def _extract_reference_expectations(path: str | Path) -> dict[int, dict[str, Any]]:
    payload = _load_json_object(path, label="reference artifact")
    summary = _get_required_dict(payload, ("summary",), label="reference artifact")
    composite_step = (
        payload.get("raw", {}).get("composite_step", {})
        if isinstance(payload.get("raw"), dict)
        else {}
    )

    expectations: dict[int, dict[str, Any]] = {}
    for batch_key, summary_entry in summary.items():
        if not isinstance(summary_entry, dict):
            continue
        batch_size = _as_int(batch_key, field_name="reference artifact batch size")
        raw_summary = {}
        if isinstance(composite_step, dict):
            raw_summary = (
                composite_step.get(str(batch_size), {})
                .get("after", {})
                .get("decode_runtime_diagnostics", {})
                .get("summary", {})
            )
        if not isinstance(raw_summary, dict):
            raw_summary = {}
        expectations[batch_size] = {
            "attention_selected_implementation": summary_entry.get("attention_impl"),
            "attention_selected_operator_pack": raw_summary.get("attention_selected_operator_pack"),
            "adjacent_selected_implementation": raw_summary.get("adjacent_selected_implementation"),
            "adjacent_selected_operator_pack": summary_entry.get("selected_pack")
            or raw_summary.get("adjacent_selected_operator_pack"),
        }

    if not expectations:
        raise ValueError("reference artifact does not expose any batch-size expectations")
    return expectations


def build_live_runtime_consistency_report(
    *,
    label: str,
    url: str,
    model: str,
    hardware_family: str,
    requested_batch_sizes: list[int],
    target_payload: dict[str, Any],
    runtime_artifacts: dict[str, str],
    reference_artifact_path: str | Path,
) -> dict[str, Any]:
    """Build a fail-fast consistency report for one live endpoint capture."""
    reference_expectations = _extract_reference_expectations(reference_artifact_path)

    info_json_path = runtime_artifacts.get("info_json")
    if not info_json_path:
        raise ValueError("runtime artifacts are missing info_json")
    info_payload = _load_json_object(info_json_path, label="info_json")
    normalized_info_payload = extract_runtime_info_payload(info_payload)
    runtime_summary = _get_required_dict(
        normalized_info_payload,
        ("performance_mainline", "decode_runtime_diagnostics", "summary"),
        label="info_json",
    )

    core_telemetry_path = runtime_artifacts.get("core_telemetry_json")
    if not core_telemetry_path:
        raise ValueError("runtime artifacts are missing core_telemetry_json")
    core_telemetry_payload = _load_json_object(core_telemetry_path, label="core_telemetry_json")
    core_summary = _get_required_dict(
        core_telemetry_payload, ("summary",), label="core_telemetry_json"
    )

    validation_batch_sizes = sorted({int(size) for size in requested_batch_sizes if int(size) <= 4})
    if not validation_batch_sizes:
        raise ValueError(
            "at least one requested batch size must be <= 4 for small-batch decode validation"
        )

    successful_batch_sizes = sorted(
        {
            int(row.get("batch_size", 0))
            for row in target_payload.get("rows", [])
            if isinstance(row, dict)
            and int(row.get("successful_requests", 0) or 0) > 0
            and int(row.get("failed_requests", 0) or 0) == 0
        }
    )
    if not any(batch_size in validation_batch_sizes for batch_size in successful_batch_sizes):
        raise ValueError(
            "live benchmark rows do not contain a successful small-batch decode scenario "
            f"for requested batch sizes {validation_batch_sizes}"
        )

    observed_batch_size = _as_int(
        runtime_summary.get("attention_batch_size"),
        field_name="performance_mainline.decode_runtime_diagnostics.summary.attention_batch_size",
    )
    if observed_batch_size not in validation_batch_sizes:
        raise ValueError(
            "runtime diagnostics batch evidence does not match requested small-batch scenario: "
            f"observed={observed_batch_size}, requested={validation_batch_sizes}"
        )
    if observed_batch_size not in successful_batch_sizes:
        raise ValueError(
            "runtime diagnostics batch evidence does not match successful live benchmark rows: "
            f"observed={observed_batch_size}, successful_rows={successful_batch_sizes}"
        )

    if observed_batch_size not in reference_expectations:
        raise ValueError(
            "reference artifact does not contain expectations for observed batch size "
            f"{observed_batch_size}"
        )
    reference = reference_expectations[observed_batch_size]

    core_by_batch_size: dict[int, dict[str, Any]] = {}
    for entry in core_summary.get("by_batch_size", []):
        if not isinstance(entry, dict):
            continue
        batch_size = _as_int(
            entry.get("batch_size"),
            field_name="core_telemetry_json.summary.by_batch_size[].batch_size",
        )
        core_by_batch_size[batch_size] = entry

    if observed_batch_size not in core_by_batch_size:
        raise ValueError(
            "core telemetry does not include the observed batch size: "
            f"observed={observed_batch_size}, available={sorted(core_by_batch_size)}"
        )
    core_batch_entry = core_by_batch_size[observed_batch_size]

    findings: list[dict[str, str]] = []

    def expect_equal(observed: Any, expected: Any, *, code: str, message: str) -> None:
        if expected is None:
            return
        if observed != expected:
            findings.append(
                {
                    "code": code,
                    "message": f"{message}: expected={expected!r}, observed={observed!r}",
                }
            )

    def expect_true(condition: bool, *, code: str, message: str) -> None:
        if not condition:
            findings.append({"code": code, "message": message})

    observed_attention_impl = runtime_summary.get("attention_selected_implementation")
    observed_attention_pack = runtime_summary.get("attention_selected_operator_pack")
    observed_adjacent_impl = runtime_summary.get("adjacent_selected_implementation")
    observed_adjacent_pack = runtime_summary.get("adjacent_selected_operator_pack")
    observed_first_failure_reason = runtime_summary.get("attention_first_failure_reason")

    expect_equal(
        observed_attention_impl,
        reference.get("attention_selected_implementation"),
        code="attention-implementation-mismatch",
        message="/info primary attention implementation disagrees with reference artifact",
    )
    expect_equal(
        observed_attention_pack,
        reference.get("attention_selected_operator_pack"),
        code="attention-pack-mismatch",
        message="/info primary attention operator pack disagrees with reference artifact",
    )
    expect_equal(
        observed_adjacent_impl,
        reference.get("adjacent_selected_implementation"),
        code="adjacent-implementation-mismatch",
        message="/info decode-adjacent implementation disagrees with reference artifact",
    )
    expect_equal(
        observed_adjacent_pack,
        reference.get("adjacent_selected_operator_pack"),
        code="adjacent-pack-mismatch",
        message="/info decode-adjacent operator pack disagrees with reference artifact",
    )

    if isinstance(observed_attention_impl, str) and observed_attention_impl.startswith("native-"):
        expect_true(
            _as_bool(runtime_summary.get("primary_decode_attention_hit")),
            code="attention-hit-missing",
            message="/info reports native primary attention but primary_decode_attention_hit is false",
        )
        expect_true(
            _as_bool(runtime_summary.get("attention_native_kernel_hit")),
            code="attention-native-hit-missing",
            message="/info reports native primary attention but attention_native_kernel_hit is false",
        )
        expect_true(
            not _as_bool(runtime_summary.get("attention_runtime_fallback")),
            code="attention-runtime-fallback",
            message="/info reports native primary attention but attention_runtime_fallback is true",
        )
        expect_true(
            observed_first_failure_reason in {None, "", "null"},
            code="attention-failure-reason-present",
            message=(
                "/info reports native primary attention but attention_first_failure_reason "
                f"is {observed_first_failure_reason!r}"
            ),
        )

    if isinstance(observed_adjacent_impl, str) and observed_adjacent_impl.startswith("native-"):
        expect_true(
            _as_bool(runtime_summary.get("adjacent_decode_pack_hit")),
            code="adjacent-hit-missing",
            message="/info reports native decode-adjacent pack but adjacent_decode_pack_hit is false",
        )
        expect_true(
            _as_bool(runtime_summary.get("adjacent_native_kernel_hit")),
            code="adjacent-native-hit-missing",
            message="/info reports native decode-adjacent pack but adjacent_native_kernel_hit is false",
        )
        expect_true(
            not _as_bool(runtime_summary.get("adjacent_runtime_fallback")),
            code="adjacent-runtime-fallback",
            message="/info reports native decode-adjacent pack but adjacent_runtime_fallback is true",
        )

    core_selected_implementations = sorted(
        str(value)
        for value in core_batch_entry.get("selected_implementations", [])
        if value is not None
    )
    core_selected_operator_packs = sorted(
        str(value)
        for value in core_batch_entry.get("selected_operator_packs", [])
        if value is not None
    )

    expect_true(
        _as_int(
            core_batch_entry.get("step_records", 0),
            field_name="core_telemetry_json.summary.by_batch_size[].step_records",
        )
        > 0,
        code="core-step-records-missing",
        message="core telemetry did not record any decode steps for the observed batch size",
    )
    expect_equal(
        core_selected_implementations,
        [observed_adjacent_impl] if observed_adjacent_impl is not None else None,
        code="core-adjacent-implementation-mismatch",
        message="core telemetry selected_implementations disagree with /info decode-adjacent implementation",
    )
    expect_equal(
        core_selected_operator_packs,
        [observed_adjacent_pack] if observed_adjacent_pack is not None else None,
        code="core-adjacent-pack-mismatch",
        message="core telemetry selected_operator_packs disagree with /info decode-adjacent operator pack",
    )

    return {
        "schema_version": "live-runtime-consistency/v1",
        "passed": not findings,
        "label": label,
        "url": url,
        "model": model,
        "hardware_family": hardware_family,
        "validation_batch_sizes": validation_batch_sizes,
        "successful_live_batch_sizes": successful_batch_sizes,
        "observed_batch_size": observed_batch_size,
        "reference_artifact": str(reference_artifact_path),
        "runtime_artifacts": dict(runtime_artifacts),
        "observed": {
            "attention_selected_implementation": observed_attention_impl,
            "attention_selected_operator_pack": observed_attention_pack,
            "attention_first_failure_reason": observed_first_failure_reason,
            "adjacent_selected_implementation": observed_adjacent_impl,
            "adjacent_selected_operator_pack": observed_adjacent_pack,
            "core_selected_implementations": core_selected_implementations,
            "core_selected_operator_packs": core_selected_operator_packs,
        },
        "reference": reference,
        "findings": findings,
    }
