from __future__ import annotations

import json

from sagellm_benchmark.runtime_consistency import (
    build_live_runtime_consistency_report,
    extract_runtime_info_payload,
)


def _write_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _reference_artifact() -> dict:
    return {
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
    }


def _info_payload() -> dict:
    return {
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
    }


def _core_telemetry_artifact() -> dict:
    return {
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
    }


def _target_payload() -> dict:
    return {
        "rows": [
            {
                "batch_size": 1,
                "successful_requests": 1,
                "failed_requests": 0,
            }
        ]
    }


def test_build_live_runtime_consistency_report_passes_for_matching_artifacts(tmp_path) -> None:
    reference_path = tmp_path / "reference.json"
    info_path = tmp_path / "sagellm_info.json"
    core_path = tmp_path / "sagellm_core_telemetry.json"
    _write_json(reference_path, _reference_artifact())
    _write_json(info_path, _info_payload())
    _write_json(core_path, _core_telemetry_artifact())

    report = build_live_runtime_consistency_report(
        label="sagellm",
        url="http://127.0.0.1:8901/v1",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_family="cuda",
        requested_batch_sizes=[1],
        target_payload=_target_payload(),
        runtime_artifacts={
            "info_json": str(info_path),
            "core_telemetry_json": str(core_path),
        },
        reference_artifact_path=reference_path,
    )

    assert report["passed"] is True
    assert report["observed_batch_size"] == 1
    assert report["reference"]["attention_selected_implementation"] == "native-cuda"
    assert report["findings"] == []


def test_build_live_runtime_consistency_report_flags_primary_attention_mismatch(tmp_path) -> None:
    reference_path = tmp_path / "reference.json"
    info_path = tmp_path / "sagellm_info.json"
    core_path = tmp_path / "sagellm_core_telemetry.json"
    info_payload = _info_payload()
    info_payload["performance_mainline"]["decode_runtime_diagnostics"]["summary"][
        "attention_selected_implementation"
    ] = "torch-fallback"
    info_payload["performance_mainline"]["decode_runtime_diagnostics"]["summary"][
        "attention_first_failure_reason"
    ] = "runtime_rejection"

    _write_json(reference_path, _reference_artifact())
    _write_json(info_path, info_payload)
    _write_json(core_path, _core_telemetry_artifact())

    report = build_live_runtime_consistency_report(
        label="sagellm",
        url="http://127.0.0.1:8901/v1",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_family="cuda",
        requested_batch_sizes=[1],
        target_payload=_target_payload(),
        runtime_artifacts={
            "info_json": str(info_path),
            "core_telemetry_json": str(core_path),
        },
        reference_artifact_path=reference_path,
    )

    assert report["passed"] is False
    assert report["findings"][0]["code"] == "attention-implementation-mismatch"


def test_extract_runtime_info_payload_accepts_gateway_info_wrapper() -> None:
    payload = {
        "service": "sageLLM Gateway",
        "registered_engines": [
            {
                "engine_id": "engine-cuda-8902",
                "info": _info_payload(),
            }
        ],
    }

    extracted = extract_runtime_info_payload(payload)

    assert extracted == _info_payload()
