from __future__ import annotations

import pytest

from sagellm_benchmark.core_telemetry import (
    build_core_decode_telemetry_artifact,
    extract_explicit_decode_snapshot,
)


def _get_info_payload() -> dict[str, object]:
    return {
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
                        "batch_id": 9,
                        "batch_type": "decode",
                        "step_index": 0,
                        "batch_size": 1,
                        "active_sequences": 1,
                        "emitted_tokens": 1,
                        "step_latency_ms": 0.5,
                        "selected_implementation": "torch-fallback",
                        "selected_operator_pack": "attention.decode",
                        "selection_interface_name": "attention_decode",
                        "telemetry_source": "step_trace",
                    },
                    {
                        "trace_id": "trace-2",
                        "request_id": "req-2",
                        "orchestration_step_id": 4,
                        "batch_id": 10,
                        "batch_type": "decode",
                        "step_index": 0,
                        "batch_size": 2,
                        "active_sequences": 2,
                        "emitted_tokens": 1,
                        "step_latency_ms": 0.8,
                        "selected_implementation": "native-paged",
                        "selected_operator_pack": "attention.decode",
                        "selection_interface_name": "attention_decode",
                        "telemetry_source": "step_trace",
                    },
                ],
                "step_telemetry_entries": 2,
                "last_orchestration_step_id": 4,
            }
        }
    }


def test_extract_explicit_decode_snapshot_accepts_get_info_payload() -> None:
    payload = _get_info_payload()

    snapshot = extract_explicit_decode_snapshot(payload)

    assert snapshot["step_telemetry_entries"] == 2


def test_extract_explicit_decode_snapshot_accepts_direct_snapshot() -> None:
    snapshot = extract_explicit_decode_snapshot(
        _get_info_payload()["performance_mainline"]["explicit_decode"]
    )

    assert snapshot["last_orchestration_step_id"] == 4


def test_build_core_decode_telemetry_artifact_normalizes_summary() -> None:
    artifact = build_core_decode_telemetry_artifact(
        _get_info_payload(),
        label="sagellm-after",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_family="cuda",
    )

    assert artifact.schema_version == "core-decode-step-telemetry/v1"
    assert artifact.step_telemetry_entries == 2
    assert artifact.summary.step_records == 2
    assert artifact.summary.batch_sizes == [1, 2]
    assert artifact.summary.selected_operator_packs == ["attention.decode"]
    assert artifact.summary.selected_implementations == ["native-paged", "torch-fallback"]
    assert artifact.summary.by_batch_size[0].batch_size == 1
    assert artifact.summary.by_batch_size[1].batch_size == 2


def test_build_core_decode_telemetry_artifact_fails_on_missing_stable_field() -> None:
    payload = _get_info_payload()
    del payload["performance_mainline"]["explicit_decode"]["step_telemetry"][0]["batch_id"]

    with pytest.raises(ValueError, match="missing stable fields"):
        build_core_decode_telemetry_artifact(
            payload,
            label="sagellm-after",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            hardware_family="cuda",
        )


def test_build_core_decode_telemetry_artifact_fails_on_entry_count_mismatch() -> None:
    payload = _get_info_payload()
    payload["performance_mainline"]["explicit_decode"]["step_telemetry_entries"] = 3

    with pytest.raises(ValueError, match="step_telemetry_entries mismatch"):
        build_core_decode_telemetry_artifact(
            payload,
            label="sagellm-after",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            hardware_family="cuda",
        )
