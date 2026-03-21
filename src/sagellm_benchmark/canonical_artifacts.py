"""Canonical benchmark artifact builders and exporters."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sagellm_benchmark.exporters import LeaderboardExporter
from sagellm_benchmark.types import AggregatedMetrics

CANONICAL_RESULT_SCHEMA_VERSION = "canonical-benchmark-result/v2"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_hardware_metadata(hardware_family: str) -> dict[str, Any]:
    normalized = hardware_family.strip().lower()
    defaults = {
        "vendor": "Unknown",
        "chip_model": hardware_family or "unknown",
        "chip_count": 1,
        "chips_per_node": 1,
        "interconnect": "None",
        "node_count": 1,
    }
    if normalized == "cuda":
        defaults.update(
            {
                "vendor": "NVIDIA",
                "chip_model": "CUDA",
            }
        )
    elif normalized == "ascend":
        defaults.update(
            {
                "vendor": "Huawei",
                "chip_model": "Ascend",
            }
        )
    elif normalized == "cpu":
        defaults.update(
            {
                "chip_model": "CPU",
            }
        )
    return defaults


def _require_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required for canonical benchmark artifacts")
    return normalized


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object in canonical benchmark artifacts")
    return value


def _coerce_metrics(metrics: AggregatedMetrics | dict[str, Any]) -> dict[str, Any]:
    if isinstance(metrics, AggregatedMetrics):
        return asdict(metrics)
    if isinstance(metrics, dict):
        return dict(metrics)
    raise TypeError(f"Unsupported metrics payload type: {type(metrics)!r}")


def _require_metric_number(metrics: dict[str, Any], field_name: str) -> float:
    value = metrics.get(field_name)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"live compare summary requires numeric field {field_name!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"live compare summary requires numeric field {field_name!r}") from exc


def _optional_metric_number(metrics: dict[str, Any], field_name: str) -> float | None:
    value = metrics.get(field_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"live compare summary field {field_name!r} must be numeric") from exc


def _optional_metric_int(metrics: dict[str, Any], field_name: str) -> int | None:
    value = metrics.get(field_name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"live compare summary field {field_name!r} must be an integer") from exc


def _build_live_compare_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    summary_payload = dict(summary)
    output_throughput_tps = _require_metric_number(summary_payload, "avg_output_throughput_tps")
    return {
        "ttft_ms": _require_metric_number(summary_payload, "avg_ttft_ms"),
        "tbt_ms": _require_metric_number(summary_payload, "avg_tbt_ms"),
        "tpot_ms": _require_metric_number(summary_payload, "avg_tpot_ms"),
        "itl_ms": _require_metric_number(summary_payload, "avg_itl_ms"),
        "p50_itl_ms": _optional_metric_number(summary_payload, "p50_itl_ms"),
        "p95_itl_ms": _optional_metric_number(summary_payload, "p95_itl_ms"),
        "p99_itl_ms": _optional_metric_number(summary_payload, "p99_itl_ms"),
        "e2el_ms": _require_metric_number(summary_payload, "avg_e2el_ms"),
        "p50_e2el_ms": _optional_metric_number(summary_payload, "p50_e2el_ms"),
        "p95_e2el_ms": _optional_metric_number(summary_payload, "p95_e2el_ms"),
        "p99_e2el_ms": _optional_metric_number(summary_payload, "p99_e2el_ms"),
        "throughput_tps": output_throughput_tps,
        "request_throughput_rps": _require_metric_number(
            summary_payload, "avg_request_throughput_rps"
        ),
        "input_throughput_tps": _optional_metric_number(summary_payload, "input_throughput_tps"),
        "output_throughput_tps": output_throughput_tps,
        "total_throughput_tps": _optional_metric_number(summary_payload, "total_throughput_tps"),
        "total_input_tokens": _optional_metric_int(summary_payload, "total_input_tokens"),
        "total_output_tokens": _optional_metric_int(summary_payload, "total_output_tokens"),
        "peak_mem_mb": int(float(summary_payload.get("peak_mem_mb") or 0.0)),
        "error_rate": float(summary_payload.get("error_rate") or 0.0),
        "prefix_hit_rate": float(summary_payload.get("prefix_hit_rate") or 0.0),
        "kv_used_tokens": int(float(summary_payload.get("kv_used_tokens") or 0.0)),
        "kv_used_bytes": int(float(summary_payload.get("kv_used_bytes") or 0.0)),
        "evict_count": int(float(summary_payload.get("evict_count") or 0.0)),
        "evict_ms": float(summary_payload.get("evict_ms") or 0.0),
        "spec_accept_rate": _optional_metric_number(summary_payload, "spec_accept_rate"),
    }


def _resolve_engine_version(label: str, versions: dict[str, Any] | None) -> str | None:
    resolved_versions = dict(versions or {})
    normalized = label.strip().lower().replace("-", "_")
    aliases = [normalized]
    if normalized == "sagellm":
        aliases.extend(["sagellm", "sagellm_benchmark", "benchmark"])
    if normalized == "vllm":
        aliases.append("vllm")
    if normalized == "vllm_ascend":
        aliases.extend(["vllm_ascend", "vllm"])
    if normalized == "lmdeploy":
        aliases.append("lmdeploy")

    for alias in aliases:
        value = resolved_versions.get(alias)
        if value:
            return str(value)
    return None


def _artifact_base(
    *,
    artifact_kind: str,
    producer_command: str,
    model: str,
    engine_name: str,
    hardware_family: str,
    workload_name: str,
    versions: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
    engine: dict[str, Any] | None,
    workload: dict[str, Any] | None,
    metrics: AggregatedMetrics | dict[str, Any],
    measurements: dict[str, Any] | None,
    telemetry: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    artifacts: dict[str, Any] | None,
    relations: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    producer_command = _require_text(producer_command, "producer_command")
    model = _require_text(model, "model")
    engine_name = _require_text(engine_name, "engine_name")
    hardware_family = _require_text(hardware_family, "hardware_family")
    workload_name = _require_text(workload_name, "workload_name")

    artifact_provenance = dict(provenance or {})
    artifact_provenance.setdefault("captured_at", _utc_now_iso())

    return {
        "schema_version": CANONICAL_RESULT_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "artifact_id": str(uuid.uuid4()),
        "producer": {
            "name": "sagellm-benchmark",
            "command": producer_command,
        },
        "provenance": artifact_provenance,
        "hardware": {
            "family": hardware_family,
            **_default_hardware_metadata(hardware_family),
        },
        "engine": {
            "name": engine_name,
            "model": model,
            **(engine or {}),
        },
        "model": {
            "name": model,
        },
        "versions": dict(versions or {}),
        "workload": {
            "name": workload_name,
            **(workload or {}),
        },
        "metrics": _coerce_metrics(metrics),
        "measurements": dict(measurements or {}),
        "telemetry": dict(telemetry or {}),
        "validation": dict(validation or {}),
        "artifacts": dict(artifacts or {}),
        "relations": list(relations or []),
    }


def build_local_run_artifact(
    *,
    workload_name: str,
    metrics: AggregatedMetrics,
    config: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = str(config.get("model_path") or config.get("model") or "unknown")
    backend = str(config.get("backend") or "cpu")
    dataset = str(config.get("dataset") or "default")
    run_id = str(config.get("run_id") or workload_name)
    return _artifact_base(
        artifact_kind="execution_result",
        producer_command="run",
        model=model,
        engine_name=backend,
        hardware_family=backend,
        workload_name=workload_name,
        versions=config.get("versions"),
        provenance={
            "captured_at": config.get("timestamp") or _utc_now_iso(),
            "run_id": run_id,
            "output_dir": str(config.get("output_dir") or ""),
        },
        engine={
            "backend": backend,
            "mode": "embedded-engine",
            "version": _resolve_engine_version(backend, config.get("versions")),
            "precision": "FP32",
        },
        workload={
            "dataset": dataset,
            "selector": config.get("workload"),
            "mode": config.get("mode", "traffic"),
            "num_samples": config.get("num_samples"),
            "precision": "FP32",
        },
        metrics=metrics,
        measurements={
            "summary": {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
            },
        },
        telemetry={},
        validation={},
        artifacts=artifacts,
        relations=[],
    )


def build_live_compare_artifact(
    *,
    label: str,
    url: str,
    model: str,
    hardware_family: str,
    batch_sizes: list[int],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    runtime_artifacts: dict[str, str] | None,
    versions: dict[str, Any] | None,
    artifacts: dict[str, Any] | None = None,
    workload_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    precision = next(
        (
            str(row.get("precision"))
            for row in rows
            if isinstance(row, dict) and row.get("precision")
        ),
        "FP16",
    )
    if precision.lower() == "live":
        precision = "FP16"

    resolved_workload_context = dict(workload_context or {})
    workload_profile = _require_text(
        resolved_workload_context.get("workload_profile", ""),
        "workload_profile",
    )
    dataset_name = _require_text(
        resolved_workload_context.get("dataset_name", ""),
        "dataset_name",
    )
    scenario_source = _require_text(
        resolved_workload_context.get("scenario_source", ""),
        "scenario_source",
    )
    supplements = resolved_workload_context.get("supplements")
    if not isinstance(supplements, list):
        raise ValueError("supplements is required for canonical benchmark artifacts")

    return _artifact_base(
        artifact_kind="execution_result",
        producer_command="compare",
        model=model,
        engine_name=label,
        hardware_family=hardware_family,
        workload_name=workload_profile,
        versions=versions,
        provenance={
            "captured_at": _utc_now_iso(),
            "endpoint_url": url,
        },
        engine={
            "endpoint": url,
            "mode": "endpoint",
            "backend": hardware_family,
            "version": _resolve_engine_version(label, versions),
            "precision": precision,
        },
        workload={
            "batch_sizes": list(batch_sizes),
            "mode": "live-compare",
            "precision": precision,
            "workload_profile": workload_profile,
            "supplements": supplements,
            "dataset_name": dataset_name,
            "scenario_source": scenario_source,
            "scenarios": sorted(
                {
                    str(row.get("scenario"))
                    for row in rows
                    if isinstance(row, dict) and row.get("scenario")
                }
            ),
        },
        metrics=_build_live_compare_metrics(summary),
        measurements={
            "summary": dict(summary),
            "rows": list(rows),
        },
        telemetry={
            "runtime_artifacts": dict(runtime_artifacts or {}),
        },
        validation={},
        artifacts=artifacts,
        relations=[],
    )


def build_compare_summary_artifact(
    *,
    model: str,
    hardware_family: str,
    batch_sizes: list[int],
    compare_result: dict[str, Any],
    target_results: list[dict[str, Any]],
    versions: dict[str, Any] | None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workload_profile = _require_text(compare_result.get("workload_profile", ""), "workload_profile")
    dataset_name = _require_text(compare_result.get("dataset_name", ""), "dataset_name")
    scenario_source = _require_text(compare_result.get("scenario_source", ""), "scenario_source")
    supplements = compare_result.get("supplements")
    if not isinstance(supplements, list):
        raise ValueError("supplements is required for compare summary canonical artifacts")

    return _artifact_base(
        artifact_kind="comparison_result",
        producer_command="compare",
        model=model,
        engine_name=str(compare_result.get("baseline") or "compare"),
        hardware_family=hardware_family,
        workload_name="compare-summary",
        versions=versions,
        provenance={
            "captured_at": _utc_now_iso(),
        },
        engine={
            "mode": "comparison",
            "baseline": compare_result.get("baseline"),
        },
        workload={
            "batch_sizes": list(batch_sizes),
            "target_labels": [str(target.get("label")) for target in target_results],
            "workload_profile": workload_profile,
            "supplements": supplements,
            "dataset_name": dataset_name,
            "scenario_source": scenario_source,
        },
        metrics={
            "target_count": len(target_results),
        },
        measurements={
            "summary": dict(compare_result),
        },
        telemetry={},
        validation={},
        artifacts=artifacts,
        relations=[
            {
                "kind": "target_artifact",
                "label": str(target.get("label")),
                "path": str(target.get("canonical_json") or ""),
            }
            for target in target_results
        ],
    )


def write_canonical_artifact(output_path: Path | str, artifact: dict[str, Any]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return path


def export_leaderboard_from_canonical_artifact(
    artifact: dict[str, Any],
    output_path: Path | str,
) -> dict[str, Any]:
    return LeaderboardExporter.export_canonical_artifact(artifact, Path(output_path))


def validate_canonical_artifact(
    artifact: dict[str, Any],
    *,
    source: Path | str | None = None,
) -> dict[str, Any]:
    source_label = str(source) if source is not None else "canonical artifact"
    payload = _require_mapping(artifact, source_label)

    schema_version = _require_text(
        payload.get("schema_version", ""), f"{source_label}.schema_version"
    )
    if schema_version != CANONICAL_RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"{source_label}.schema_version must be {CANONICAL_RESULT_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    _require_text(payload.get("artifact_kind", ""), f"{source_label}.artifact_kind")
    _require_text(payload.get("artifact_id", ""), f"{source_label}.artifact_id")

    producer = _require_mapping(payload.get("producer"), f"{source_label}.producer")
    _require_text(producer.get("name", ""), f"{source_label}.producer.name")
    _require_text(producer.get("command", ""), f"{source_label}.producer.command")

    hardware = _require_mapping(payload.get("hardware"), f"{source_label}.hardware")
    _require_text(hardware.get("family", ""), f"{source_label}.hardware.family")

    engine = _require_mapping(payload.get("engine"), f"{source_label}.engine")
    _require_text(engine.get("name", ""), f"{source_label}.engine.name")
    _require_text(engine.get("model", ""), f"{source_label}.engine.model")

    model = _require_mapping(payload.get("model"), f"{source_label}.model")
    _require_text(model.get("name", ""), f"{source_label}.model.name")

    workload = _require_mapping(payload.get("workload"), f"{source_label}.workload")
    _require_text(workload.get("name", ""), f"{source_label}.workload.name")

    _require_mapping(payload.get("metrics"), f"{source_label}.metrics")
    _require_mapping(payload.get("measurements"), f"{source_label}.measurements")
    _require_mapping(payload.get("telemetry"), f"{source_label}.telemetry")
    _require_mapping(payload.get("validation"), f"{source_label}.validation")
    _require_mapping(payload.get("artifacts"), f"{source_label}.artifacts")

    relations = payload.get("relations")
    if not isinstance(relations, list):
        raise ValueError(
            f"{source_label}.relations must be a list in canonical benchmark artifacts"
        )

    return payload


def load_canonical_artifact(input_path: Path | str) -> dict[str, Any]:
    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_canonical_artifact(payload, source=path)


def collect_canonical_artifacts(
    input_dir: Path | str,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    input_path = Path(input_dir)
    errors: list[str] = []
    artifacts: list[tuple[Path, dict[str, Any]]] = []

    canonical_paths = sorted(input_path.rglob("*.canonical.json"))
    if not canonical_paths:
        errors.append(f"No *.canonical.json found under: {input_path}")
        return artifacts, errors

    for canonical_path in canonical_paths:
        try:
            artifact = load_canonical_artifact(canonical_path)
        except Exception as exc:
            errors.append(f"{canonical_path}: {exc}")
            continue
        artifacts.append((canonical_path, artifact))

    return artifacts, errors


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = [float(row.get("ttft_ms", 0.0)) for row in rows]
    tbt = [float(row.get("tbt_ms", 0.0)) for row in rows]
    tpot = [float(row.get("tpot_ms", 0.0)) for row in rows]
    itl = [float(row.get("avg_itl_ms", 0.0)) for row in rows]
    e2el = [float(row.get("avg_e2el_ms", 0.0)) for row in rows]
    throughput = [float(row.get("throughput_tps", 0.0)) for row in rows]
    output_throughput = [
        float(row.get("output_throughput_tps", row.get("throughput_tps", 0.0))) for row in rows
    ]
    request_throughput = [float(row.get("request_throughput_rps", 0.0)) for row in rows]
    success = sum(int(row.get("successful_requests", 0)) for row in rows)
    failed = sum(int(row.get("failed_requests", 0)) for row in rows)
    total = success + failed
    return {
        "ttft_ms": _mean(ttft),
        "tbt_ms": _mean(tbt),
        "tpot_ms": _mean(tpot),
        "itl_ms": _mean(itl),
        "e2el_ms": _mean(e2el),
        "throughput_tps": _mean(throughput),
        "output_throughput_tps": _mean(output_throughput),
        "request_throughput_rps": _mean(request_throughput),
        "peak_mem_mb": 0,
        "error_rate": (failed / total) if total else 0.0,
        "prefix_hit_rate": 0.0,
        "kv_used_tokens": 0,
        "kv_used_bytes": 0,
        "evict_count": 0,
        "evict_ms": 0.0,
    }


def _prepare_leaderboard_export_artifact(
    artifact: dict[str, Any],
    *,
    include_supplements: bool,
) -> dict[str, Any]:
    if include_supplements:
        return artifact

    workload = artifact.get("workload") if isinstance(artifact.get("workload"), dict) else {}
    supplements = workload.get("supplements") if isinstance(workload, dict) else []
    if not supplements:
        return artifact

    measurements = (
        artifact.get("measurements") if isinstance(artifact.get("measurements"), dict) else {}
    )
    rows = measurements.get("rows") if isinstance(measurements.get("rows"), list) else []
    mainline_rows = [
        row for row in rows if isinstance(row, dict) and row.get("scenario_source") == "mainline"
    ]
    if not mainline_rows:
        raise ValueError("supplement export requires at least one mainline scenario row")

    cloned = json.loads(json.dumps(artifact))
    cloned["measurements"]["rows"] = mainline_rows
    cloned["workload"]["supplements"] = []
    cloned["workload"]["scenario_source"] = "mainline"
    cloned["metrics"] = _metrics_from_rows(mainline_rows)
    return cloned


def export_standard_leaderboard_artifacts(
    input_dir: Path | str,
    *,
    include_supplements: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    artifacts, errors = collect_canonical_artifacts(input_path)
    if errors:
        raise ValueError("\n".join(errors))

    manifest_path = input_path / "leaderboard_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    exported: list[dict[str, Any]] = []
    for canonical_path, artifact in artifacts:
        if artifact.get("artifact_kind") != "execution_result":
            continue
        producer = artifact.get("producer") if isinstance(artifact.get("producer"), dict) else {}
        workload = artifact.get("workload") if isinstance(artifact.get("workload"), dict) else {}
        if producer.get("command") != "compare" or workload.get("mode") != "live-compare":
            continue

        file_stem = canonical_path.name.removesuffix(".canonical.json")
        leaderboard_path = canonical_path.with_name(f"{file_stem}_leaderboard.json")
        export_artifact = _prepare_leaderboard_export_artifact(
            artifact,
            include_supplements=include_supplements,
        )
        leaderboard_entry = export_leaderboard_from_canonical_artifact(
            export_artifact,
            leaderboard_path,
        )
        artifact.setdefault("artifacts", {})["leaderboard_json"] = str(leaderboard_path)
        write_canonical_artifact(canonical_path, artifact)
        LeaderboardExporter.register_exported_entry(
            output_dir=input_path,
            entry=leaderboard_entry,
            leaderboard_path=leaderboard_path,
            canonical_artifact_path=canonical_path,
        )
        exported.append(
            {
                "canonical_artifact": str(canonical_path),
                "leaderboard_artifact": str(leaderboard_path),
                "entry_id": leaderboard_entry["entry_id"],
                "idempotency_key": leaderboard_entry["metadata"]["idempotency_key"],
            }
        )

    if not exported:
        raise ValueError(
            f"No canonical live-compare execution_result artifacts found under: {input_path}"
        )

    return {
        "validated_count": len(artifacts),
        "exported_count": len(exported),
        "manifest_path": str(manifest_path),
        "exports": exported,
    }
