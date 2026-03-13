"""Stable benchmark-side consumers for sagellm-core decode step telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _CompatModel(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=False)


class CoreDecodeStepTelemetryRecord(_CompatModel):
    """One normalized decode-step telemetry record exported by sagellm-core."""

    trace_id: str
    request_id: str
    orchestration_step_id: int = Field(ge=0)
    batch_id: int = Field(ge=0)
    batch_type: str
    step_index: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    active_sequences: int = Field(ge=0)
    emitted_tokens: int = Field(ge=0)
    step_latency_ms: float = Field(ge=0.0)
    selected_implementation: str
    selected_operator_pack: str
    selection_interface_name: str
    telemetry_source: str


class CoreDecodeBatchSummary(_CompatModel):
    """Per-batch-size telemetry summary for before/after comparisons."""

    batch_size: int = Field(ge=1)
    step_records: int = Field(ge=0)
    unique_requests: int = Field(ge=0)
    avg_step_latency_ms: float = Field(ge=0.0)
    max_step_latency_ms: float = Field(ge=0.0)
    selected_implementations: list[str] = Field(default_factory=list)
    selected_operator_packs: list[str] = Field(default_factory=list)


class CoreDecodeTelemetrySummary(_CompatModel):
    """Top-level summary over normalized core decode telemetry."""

    step_records: int = Field(ge=0)
    unique_requests: int = Field(ge=0)
    batch_sizes: list[int] = Field(default_factory=list)
    selected_implementations: list[str] = Field(default_factory=list)
    selected_operator_packs: list[str] = Field(default_factory=list)
    avg_step_latency_ms: float = Field(ge=0.0)
    max_step_latency_ms: float = Field(ge=0.0)
    by_batch_size: list[CoreDecodeBatchSummary] = Field(default_factory=list)


class CoreDecodeTelemetryArtifact(_CompatModel):
    """Reusable benchmark artifact derived from LLMEngine.get_info() explicit decode telemetry."""

    schema_version: str = Field(default="core-decode-step-telemetry/v1")
    label: str
    model: str
    hardware_family: str
    feature_gate: dict[str, Any]
    step_telemetry_schema_version: int = Field(ge=1)
    step_telemetry_stable_fields: list[str] = Field(default_factory=list)
    step_telemetry_entries: int = Field(ge=0)
    last_orchestration_step_id: int = Field(ge=0)
    step_telemetry: list[CoreDecodeStepTelemetryRecord] = Field(default_factory=list)
    summary: CoreDecodeTelemetrySummary
    metadata: dict[str, Any] | None = Field(default=None)


def load_core_decode_telemetry_input(path: str | Path) -> dict[str, Any]:
    """Load a core telemetry input from disk.

    Accepts either the full `LLMEngine.get_info()` JSON payload or the nested
    `performance_mainline.explicit_decode` object serialized on its own.
    """
    payload_path = Path(path)
    with payload_path.open() as handle:
        payload = json.load(handle)
    return extract_explicit_decode_snapshot(payload)


def extract_explicit_decode_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the explicit decode snapshot from a full `get_info()` payload or direct snapshot."""
    if not isinstance(payload, dict):
        raise ValueError("Core telemetry payload must be a JSON object")

    if "performance_mainline" in payload:
        performance_mainline = payload.get("performance_mainline")
        if not isinstance(performance_mainline, dict):
            raise ValueError("Core telemetry payload performance_mainline must be an object")
        explicit_decode = performance_mainline.get("explicit_decode")
        if not isinstance(explicit_decode, dict):
            raise ValueError(
                "Core telemetry payload is missing performance_mainline.explicit_decode"
            )
        return explicit_decode

    if "feature_gate" in payload and "step_telemetry" in payload:
        return payload

    raise ValueError(
        "Core telemetry payload must be a full get_info() dump or an explicit_decode snapshot"
    )


def build_core_decode_telemetry_artifact(
    payload: dict[str, Any],
    *,
    label: str,
    model: str,
    hardware_family: str,
) -> CoreDecodeTelemetryArtifact:
    """Convert a core explicit decode snapshot into a stable benchmark artifact."""
    snapshot = extract_explicit_decode_snapshot(payload)

    required_top_level_fields = (
        "feature_gate",
        "step_telemetry_schema_version",
        "step_telemetry_stable_fields",
        "step_telemetry",
        "step_telemetry_entries",
        "last_orchestration_step_id",
    )
    missing_top_level_fields = [
        field_name for field_name in required_top_level_fields if field_name not in snapshot
    ]
    if missing_top_level_fields:
        raise ValueError(
            f"Core explicit decode snapshot missing stable fields: {missing_top_level_fields}"
        )

    stable_fields = snapshot["step_telemetry_stable_fields"]
    if not isinstance(stable_fields, list) or not stable_fields:
        raise ValueError(
            "Core explicit decode snapshot requires non-empty step_telemetry_stable_fields"
        )

    raw_step_telemetry = snapshot["step_telemetry"]
    if not isinstance(raw_step_telemetry, list):
        raise ValueError("Core explicit decode snapshot step_telemetry must be a list")
    if snapshot["step_telemetry_entries"] != len(raw_step_telemetry):
        raise ValueError(
            "Core explicit decode snapshot step_telemetry_entries mismatch: "
            f"expected {len(raw_step_telemetry)}, got {snapshot['step_telemetry_entries']}"
        )

    normalized_step_telemetry: list[CoreDecodeStepTelemetryRecord] = []
    for index, entry in enumerate(raw_step_telemetry):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Core explicit decode telemetry entry must be an object: index={index}"
            )
        missing_entry_fields = [
            field_name for field_name in stable_fields if field_name not in entry
        ]
        if missing_entry_fields:
            raise ValueError(
                "Core explicit decode telemetry entry missing stable fields: "
                f"index={index}, missing={missing_entry_fields}"
            )
        normalized_step_telemetry.append(CoreDecodeStepTelemetryRecord.model_validate(entry))

    summary = _summarize_step_telemetry(normalized_step_telemetry)
    return CoreDecodeTelemetryArtifact(
        label=label,
        model=model,
        hardware_family=hardware_family,
        feature_gate=dict(snapshot["feature_gate"]),
        step_telemetry_schema_version=int(snapshot["step_telemetry_schema_version"]),
        step_telemetry_stable_fields=list(stable_fields),
        step_telemetry_entries=int(snapshot["step_telemetry_entries"]),
        last_orchestration_step_id=int(snapshot["last_orchestration_step_id"]),
        step_telemetry=normalized_step_telemetry,
        summary=summary,
        metadata={
            "source": "sagellm_core.performance_mainline.explicit_decode",
        },
    )


def _summarize_step_telemetry(
    step_telemetry: list[CoreDecodeStepTelemetryRecord],
) -> CoreDecodeTelemetrySummary:
    """Build a compact summary over normalized decode-step telemetry rows."""
    if not step_telemetry:
        return CoreDecodeTelemetrySummary()

    batch_sizes = sorted({entry.batch_size for entry in step_telemetry})
    selected_implementations = sorted({entry.selected_implementation for entry in step_telemetry})
    selected_operator_packs = sorted({entry.selected_operator_pack for entry in step_telemetry})
    unique_requests = len({entry.request_id for entry in step_telemetry})
    avg_step_latency_ms = sum(entry.step_latency_ms for entry in step_telemetry) / len(
        step_telemetry
    )
    max_step_latency_ms = max(entry.step_latency_ms for entry in step_telemetry)

    by_batch_size: list[CoreDecodeBatchSummary] = []
    for batch_size in batch_sizes:
        batch_entries = [entry for entry in step_telemetry if entry.batch_size == batch_size]
        by_batch_size.append(
            CoreDecodeBatchSummary(
                batch_size=batch_size,
                step_records=len(batch_entries),
                unique_requests=len({entry.request_id for entry in batch_entries}),
                avg_step_latency_ms=(
                    sum(entry.step_latency_ms for entry in batch_entries) / len(batch_entries)
                ),
                max_step_latency_ms=max(entry.step_latency_ms for entry in batch_entries),
                selected_implementations=sorted(
                    {entry.selected_implementation for entry in batch_entries}
                ),
                selected_operator_packs=sorted(
                    {entry.selected_operator_pack for entry in batch_entries}
                ),
            )
        )

    return CoreDecodeTelemetrySummary(
        step_records=len(step_telemetry),
        unique_requests=unique_requests,
        batch_sizes=batch_sizes,
        selected_implementations=selected_implementations,
        selected_operator_packs=selected_operator_packs,
        avg_step_latency_ms=avg_step_latency_ms,
        max_step_latency_ms=max_step_latency_ms,
        by_batch_size=by_batch_size,
    )
