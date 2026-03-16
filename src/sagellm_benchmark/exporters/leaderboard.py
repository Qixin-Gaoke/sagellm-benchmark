"""Export benchmark results to sageLLM website leaderboard format."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sagellm_benchmark.workloads import M1_WORKLOADS, TPCH_WORKLOADS

if TYPE_CHECKING:
    from sagellm_benchmark.types import AggregatedMetrics


WORKLOAD_SPECS = {
    workload.name: {
        "input_length": workload.prompt_tokens,
        "output_length": workload.max_tokens,
        "batch_size": 1,
        "concurrent_requests": workload.concurrency
        or (workload.num_requests if workload.concurrent else 1),
    }
    for workload in [*TPCH_WORKLOADS, *M1_WORKLOADS]
}

LEADERBOARD_ENTRY_SCHEMA_VERSION = "leaderboard-export-entry/v2"
LEADERBOARD_MANIFEST_SCHEMA_VERSION = "leaderboard-export-manifest/v2"
LEADERBOARD_COMPARE_SNAPSHOT_SCHEMA_VERSION = "leaderboard-compare-snapshot/v1"
SUPPORTED_CANONICAL_EXPORT_SCHEMA_VERSION = "canonical-benchmark-result/v2"


class LeaderboardExporter:
    """Export benchmark results to leaderboard format."""

    _REQUIRED_METRIC_FIELDS = (
        "throughput_tps",
        "peak_mem_mb",
        "error_rate",
        "prefix_hit_rate",
        "kv_used_tokens",
        "kv_used_bytes",
        "evict_count",
        "evict_ms",
    )

    _OPTIONAL_METRIC_FIELDS = (
        "ttft_ms",
        "tbt_ms",
        "tpot_ms",
        "itl_ms",
        "p50_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "e2el_ms",
        "p50_e2el_ms",
        "p95_e2el_ms",
        "p99_e2el_ms",
        "request_throughput_rps",
        "input_throughput_tps",
        "output_throughput_tps",
        "total_throughput_tps",
        "total_input_tokens",
        "total_output_tokens",
        "spec_accept_rate",
    )

    @staticmethod
    def _require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object")
        return value

    @staticmethod
    def _require_text(value: Any, *, field_name: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{field_name} is required")
        return text

    @staticmethod
    def _require_number(value: Any, *, field_name: str) -> float:
        if isinstance(value, bool) or value is None:
            raise ValueError(f"{field_name} must be numeric")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric") from exc

    @staticmethod
    def _validate_optional_number(value: Any, *, field_name: str) -> float | None:
        if value is None:
            return None
        return LeaderboardExporter._require_number(value, field_name=field_name)

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _entry_bucket(entry: dict[str, Any]) -> str:
        cluster = entry.get("cluster") or {}
        node_count = int(cluster.get("node_count") or 1)
        return "multi" if node_count > 1 else "single"

    @staticmethod
    def validate_leaderboard_entry(
        entry: dict[str, Any], *, label: str = "leaderboard entry"
    ) -> dict[str, Any]:
        payload = LeaderboardExporter._require_mapping(entry, field_name=label)

        schema_version = LeaderboardExporter._require_text(
            payload.get("schema_version"), field_name=f"{label}.schema_version"
        )
        if schema_version != LEADERBOARD_ENTRY_SCHEMA_VERSION:
            raise ValueError(f"{label}.schema_version must be {LEADERBOARD_ENTRY_SCHEMA_VERSION!r}")

        LeaderboardExporter._require_text(payload.get("entry_id"), field_name=f"{label}.entry_id")
        LeaderboardExporter._require_text(payload.get("engine"), field_name=f"{label}.engine")
        LeaderboardExporter._require_text(
            payload.get("engine_version"), field_name=f"{label}.engine_version"
        )
        LeaderboardExporter._require_text(
            payload.get("sagellm_version"), field_name=f"{label}.sagellm_version"
        )

        config_type = LeaderboardExporter._require_text(
            payload.get("config_type"), field_name=f"{label}.config_type"
        )
        if config_type not in {"single_gpu", "multi_gpu", "multi_node"}:
            raise ValueError(f"{label}.config_type must be single_gpu, multi_gpu, or multi_node")

        hardware = LeaderboardExporter._require_mapping(
            payload.get("hardware"), field_name=f"{label}.hardware"
        )
        LeaderboardExporter._require_text(
            hardware.get("chip_model"), field_name=f"{label}.hardware.chip_model"
        )
        LeaderboardExporter._require_number(
            hardware.get("chip_count"), field_name=f"{label}.hardware.chip_count"
        )
        LeaderboardExporter._require_text(
            hardware.get("interconnect"), field_name=f"{label}.hardware.interconnect"
        )

        model = LeaderboardExporter._require_mapping(
            payload.get("model"), field_name=f"{label}.model"
        )
        LeaderboardExporter._require_text(model.get("name"), field_name=f"{label}.model.name")
        LeaderboardExporter._require_text(
            model.get("precision"), field_name=f"{label}.model.precision"
        )

        workload = LeaderboardExporter._require_mapping(
            payload.get("workload"), field_name=f"{label}.workload"
        )
        LeaderboardExporter._require_text(workload.get("name"), field_name=f"{label}.workload.name")
        LeaderboardExporter._require_number(
            workload.get("input_length"), field_name=f"{label}.workload.input_length"
        )
        LeaderboardExporter._require_number(
            workload.get("output_length"), field_name=f"{label}.workload.output_length"
        )

        metrics = LeaderboardExporter._require_mapping(
            payload.get("metrics"), field_name=f"{label}.metrics"
        )
        for field in LeaderboardExporter._REQUIRED_METRIC_FIELDS:
            LeaderboardExporter._require_number(
                metrics.get(field), field_name=f"{label}.metrics.{field}"
            )
        for field in LeaderboardExporter._OPTIONAL_METRIC_FIELDS:
            LeaderboardExporter._validate_optional_number(
                metrics.get(field), field_name=f"{label}.metrics.{field}"
            )

        versions = LeaderboardExporter._require_mapping(
            payload.get("versions"), field_name=f"{label}.versions"
        )
        LeaderboardExporter._require_text(
            versions.get("benchmark"), field_name=f"{label}.versions.benchmark"
        )

        environment = LeaderboardExporter._require_mapping(
            payload.get("environment"), field_name=f"{label}.environment"
        )
        LeaderboardExporter._require_text(
            environment.get("os"), field_name=f"{label}.environment.os"
        )
        LeaderboardExporter._require_text(
            environment.get("python_version"), field_name=f"{label}.environment.python_version"
        )

        metadata = LeaderboardExporter._require_mapping(
            payload.get("metadata"), field_name=f"{label}.metadata"
        )
        LeaderboardExporter._require_text(
            metadata.get("submitted_at"), field_name=f"{label}.metadata.submitted_at"
        )
        LeaderboardExporter._require_text(
            metadata.get("engine"), field_name=f"{label}.metadata.engine"
        )
        LeaderboardExporter._require_text(
            metadata.get("engine_version"), field_name=f"{label}.metadata.engine_version"
        )
        LeaderboardExporter._require_text(
            metadata.get("hardware_family"), field_name=f"{label}.metadata.hardware_family"
        )

        cluster = payload.get("cluster")
        if cluster is not None:
            cluster_payload = LeaderboardExporter._require_mapping(
                cluster, field_name=f"{label}.cluster"
            )
            node_count = int(
                LeaderboardExporter._require_number(
                    cluster_payload.get("node_count"), field_name=f"{label}.cluster.node_count"
                )
            )
            if node_count > 1 and config_type != "multi_node":
                raise ValueError(
                    f"{label}.config_type must be multi_node when cluster.node_count > 1"
                )

        return payload

    @staticmethod
    def detect_hardware_info() -> dict[str, Any]:
        cpu_model = platform.processor() or "unknown"
        if not cpu_model or cpu_model == "unknown":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_model = line.split(":", 1)[1].strip()
                            break
            except OSError:
                pass
            if not cpu_model or cpu_model == "unknown":
                cpu_model = platform.machine() or "CPU"

        cpu_vendor = "Unknown"
        cpu_lower = cpu_model.lower()
        if "intel" in cpu_lower:
            cpu_vendor = "Intel"
        elif "amd" in cpu_lower or "ryzen" in cpu_lower or "epyc" in cpu_lower:
            cpu_vendor = "AMD"
        elif "apple" in cpu_lower or "m1" in cpu_lower or "m2" in cpu_lower or "m3" in cpu_lower:
            cpu_vendor = "Other"
        elif "arm" in cpu_lower or "aarch" in cpu_lower:
            cpu_vendor = "Other"

        total_memory_gb = 0.0
        try:
            import psutil

            total_memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            total_memory_gb = round(kb / (1024**2), 2)
                            break
            except OSError:
                pass

        hardware = {
            "vendor": cpu_vendor,
            "chip_model": cpu_model,
            "chip_count": 1,
            "chips_per_node": 1,
            "interconnect": "None",
            "intra_node_interconnect": "None",
            "memory_per_chip_gb": total_memory_gb,
            "total_memory_gb": total_memory_gb,
            "node_count": 1,
            "family": "cpu",
        }

        try:
            import torch

            if torch.cuda.is_available():
                chip_count = torch.cuda.device_count()
                per_chip_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                hardware.update(
                    {
                        "vendor": "NVIDIA",
                        "chip_model": torch.cuda.get_device_name(0),
                        "chip_count": chip_count,
                        "chips_per_node": chip_count,
                        "interconnect": "NVLink" if chip_count > 1 else "None",
                        "intra_node_interconnect": "NVLink" if chip_count > 1 else "None",
                        "memory_per_chip_gb": per_chip_gb,
                        "total_memory_gb": round(per_chip_gb * chip_count, 2),
                        "family": "cuda",
                    }
                )
        except Exception:
            pass

        try:
            import torch_npu

            if torch_npu.npu.is_available():
                chip_count = torch_npu.npu.device_count()
                hardware.update(
                    {
                        "vendor": "Huawei",
                        "chip_model": "Ascend 910B",
                        "chip_count": chip_count,
                        "chips_per_node": chip_count,
                        "interconnect": "HCCS" if chip_count > 1 else "None",
                        "intra_node_interconnect": "HCCS" if chip_count > 1 else "None",
                        "family": "ascend",
                    }
                )
        except Exception:
            pass

        return hardware

    @staticmethod
    def detect_environment() -> dict[str, Any]:
        env = {
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "pytorch_version": "",
            "cuda_version": "",
            "cann_version": "",
            "driver_version": "",
        }

        try:
            import torch

            env["pytorch_version"] = torch.__version__
            if torch.cuda.is_available():
                env["cuda_version"] = torch.version.cuda
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                env["driver_version"] = result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return env

    @staticmethod
    def infer_config_type(chip_count: int, has_cluster: bool) -> str:
        if has_cluster:
            return "multi_node"
        if chip_count > 1:
            return "multi_gpu"
        return "single_gpu"

    @staticmethod
    def _resolve_version(
        versions: dict[str, Any],
        aliases: tuple[str, ...],
        default: str = "N/A",
    ) -> str:
        for key in aliases:
            value = versions.get(key)
            if value:
                return str(value)
        return default

    @staticmethod
    def _normalize_engine_name(raw_value: Any) -> str:
        raw = str(raw_value or "").strip().lower()
        if not raw:
            return "unknown"
        return raw.replace(" ", "-")

    @staticmethod
    def _normalize_precision(raw_value: Any, *, default: str = "FP32") -> str:
        normalized = str(raw_value or "").strip().lower()
        mapping = {
            "fp32": "FP32",
            "float32": "FP32",
            "fp16": "FP16",
            "float16": "FP16",
            "half": "FP16",
            "bf16": "BF16",
            "bfloat16": "BF16",
            "int8": "INT8",
            "int4": "INT4",
            "fp8": "FP8",
            "live": default,
            "": default,
        }
        return mapping.get(normalized, default)

    @staticmethod
    def _normalize_vendor(raw_value: Any) -> str:
        normalized = str(raw_value or "Unknown").strip()
        allowed = {"Intel", "AMD", "NVIDIA", "Huawei", "Unknown", "Other"}
        return normalized if normalized in allowed else "Other"

    @staticmethod
    def _extract_workload_spec(workload_name: str) -> dict[str, Any]:
        return WORKLOAD_SPECS.get(
            workload_name,
            {
                "input_length": 128,
                "output_length": 128,
                "batch_size": None,
                "concurrent_requests": None,
            },
        )

    @staticmethod
    def _extract_workload_name_from_entry(entry: dict[str, Any]) -> str:
        workload = entry.get("workload") or {}
        if isinstance(workload, dict):
            for key in ("name", "workload_id", "suite_id"):
                value = workload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        notes = str(entry.get("metadata", {}).get("notes") or "")
        q_match = re.search(r"\bQ([1-8])\b", notes, flags=re.IGNORECASE)
        if q_match:
            return f"Q{q_match.group(1)}"
        return "LEGACY"

    @staticmethod
    def _build_compare_scope_key(entry: dict[str, Any]) -> str:
        workload_name = LeaderboardExporter._extract_workload_name_from_entry(entry)
        hardware = str(entry.get("hardware", {}).get("chip_model") or "unknown-hardware")
        model = str(entry.get("model", {}).get("name") or "unknown-model")
        precision = str(entry.get("model", {}).get("precision") or "unknown-precision")
        config_type = str(entry.get("config_type") or "unknown-config")
        chip_count = int(entry.get("hardware", {}).get("chip_count") or 0)
        node_count = int((entry.get("cluster") or {}).get("node_count") or 1)
        return "|".join(
            [
                model,
                hardware,
                precision,
                workload_name,
                config_type,
                str(chip_count),
                str(node_count),
            ]
        )

    @staticmethod
    def _engine_sort_key(entry: dict[str, Any]) -> tuple[int, str, str]:
        engine = str(entry.get("engine") or "unknown")
        priority = {
            "sagellm": 0,
            "vllm": 1,
            "vllm-ascend": 2,
            "sglang": 3,
            "lmdeploy": 4,
        }.get(engine, 10)
        version = str(entry.get("engine_version") or "")
        return (priority, engine, version)

    @staticmethod
    def _build_compare_engine_summary(entry: dict[str, Any]) -> dict[str, Any]:
        metrics = entry.get("metrics") or {}
        metadata = entry.get("metadata") or {}
        return {
            "engine": str(entry.get("engine") or "unknown"),
            "engine_version": str(entry.get("engine_version") or "unknown"),
            "entry_id": str(entry.get("entry_id") or ""),
            "submitted_at": metadata.get("submitted_at"),
            "canonical_path": entry.get("canonical_path"),
            "metrics": {
                "ttft_ms": float(metrics.get("ttft_ms") or 0.0),
                "tbt_ms": float(metrics.get("tbt_ms") or 0.0),
                "throughput_tps": float(metrics.get("throughput_tps") or 0.0),
                "output_throughput_tps": float(
                    metrics.get("output_throughput_tps") or metrics.get("throughput_tps") or 0.0
                ),
                "error_rate": float(metrics.get("error_rate") or 0.0),
            },
        }

    @staticmethod
    def _compute_relative_delta(
        left_value: float | int | None,
        right_value: float | int | None,
    ) -> float | None:
        if left_value is None or right_value in (None, 0):
            return None
        return round(
            ((float(left_value) - float(right_value)) / abs(float(right_value))) * 100.0, 4
        )

    @staticmethod
    def _metric_winner(
        left_value: float | int | None,
        right_value: float | int | None,
        *,
        higher_is_better: bool,
    ) -> str:
        if left_value is None or right_value is None:
            return "unknown"
        if float(left_value) == float(right_value):
            return "parity"
        if higher_is_better:
            return "left" if float(left_value) > float(right_value) else "right"
        return "left" if float(left_value) < float(right_value) else "right"

    @staticmethod
    def _select_preferred_pair(
        entries: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        by_engine = {str(entry.get("engine") or "unknown"): entry for entry in entries}
        preferred_orders = [
            ("sagellm", "vllm"),
            ("sagellm", "vllm-ascend"),
            ("sagellm", "sglang"),
            ("sagellm", "lmdeploy"),
        ]
        for left_engine, right_engine in preferred_orders:
            if left_engine in by_engine and right_engine in by_engine:
                return by_engine[left_engine], by_engine[right_engine]

        if len(entries) >= 2:
            ordered = sorted(entries, key=LeaderboardExporter._engine_sort_key)
            return ordered[0], ordered[1]
        return None

    @staticmethod
    def build_compare_snapshot(entries: list[dict[str, Any]]) -> dict[str, Any]:
        deduped_entries: dict[str, dict[str, Any]] = {}
        for entry in entries:
            normalized = LeaderboardExporter.annotate_entry_identity(entry)
            key = normalized["metadata"]["idempotency_key"]
            existing = deduped_entries.get(key)
            deduped_entries[key] = (
                LeaderboardExporter.prefer_newer_entry(existing, normalized)
                if existing is not None
                else normalized
            )

        grouped: dict[str, dict[str, Any]] = {}
        for entry in deduped_entries.values():
            scope_key = LeaderboardExporter._build_compare_scope_key(entry)
            group = grouped.setdefault(
                scope_key,
                {
                    "scope_key": scope_key,
                    "scope": {
                        "model": str(entry.get("model", {}).get("name") or "unknown-model"),
                        "hardware": str(
                            entry.get("hardware", {}).get("chip_model") or "unknown-hardware"
                        ),
                        "precision": str(
                            entry.get("model", {}).get("precision") or "unknown-precision"
                        ),
                        "workload": LeaderboardExporter._extract_workload_name_from_entry(entry),
                        "config_type": str(entry.get("config_type") or "unknown-config"),
                        "chip_count": int(entry.get("hardware", {}).get("chip_count") or 0),
                        "node_count": int((entry.get("cluster") or {}).get("node_count") or 1),
                    },
                    "category": LeaderboardExporter._entry_bucket(entry),
                    "entries_by_engine": {},
                },
            )

            engine = str(entry.get("engine") or "unknown")
            existing = group["entries_by_engine"].get(engine)
            group["entries_by_engine"][engine] = (
                LeaderboardExporter.prefer_newer_entry(existing, entry)
                if existing is not None
                else entry
            )

        groups_payload: list[dict[str, Any]] = []
        preferred_pairs: list[dict[str, Any]] = []
        for group in grouped.values():
            representative_entries = sorted(
                group["entries_by_engine"].values(), key=LeaderboardExporter._engine_sort_key
            )
            if len(representative_entries) < 2:
                continue

            preferred_pair = LeaderboardExporter._select_preferred_pair(representative_entries)
            if preferred_pair is None:
                continue

            left_entry, right_entry = preferred_pair
            left_summary = LeaderboardExporter._build_compare_engine_summary(left_entry)
            right_summary = LeaderboardExporter._build_compare_engine_summary(right_entry)
            pair_payload = {
                "left": left_summary,
                "right": right_summary,
                "deltas": {
                    "throughput_pct_left_vs_right": LeaderboardExporter._compute_relative_delta(
                        left_summary["metrics"]["throughput_tps"],
                        right_summary["metrics"]["throughput_tps"],
                    ),
                    "ttft_pct_left_vs_right": LeaderboardExporter._compute_relative_delta(
                        left_summary["metrics"]["ttft_ms"],
                        right_summary["metrics"]["ttft_ms"],
                    ),
                    "tbt_pct_left_vs_right": LeaderboardExporter._compute_relative_delta(
                        left_summary["metrics"]["tbt_ms"],
                        right_summary["metrics"]["tbt_ms"],
                    ),
                },
                "winners": {
                    "throughput": LeaderboardExporter._metric_winner(
                        left_summary["metrics"]["throughput_tps"],
                        right_summary["metrics"]["throughput_tps"],
                        higher_is_better=True,
                    ),
                    "ttft": LeaderboardExporter._metric_winner(
                        left_summary["metrics"]["ttft_ms"],
                        right_summary["metrics"]["ttft_ms"],
                        higher_is_better=False,
                    ),
                    "tbt": LeaderboardExporter._metric_winner(
                        left_summary["metrics"]["tbt_ms"],
                        right_summary["metrics"]["tbt_ms"],
                        higher_is_better=False,
                    ),
                },
            }

            group_payload = {
                "scope_key": group["scope_key"],
                "category": group["category"],
                "scope": group["scope"],
                "engines": [
                    LeaderboardExporter._build_compare_engine_summary(entry)
                    for entry in representative_entries
                ],
                "preferred_pair": pair_payload,
            }
            groups_payload.append(group_payload)

            if left_summary["engine"] == "sagellm" and right_summary["engine"] in {
                "vllm",
                "vllm-ascend",
            }:
                preferred_pairs.append(group_payload)

        groups_payload.sort(key=lambda item: str(item["scope_key"]))
        preferred_pairs.sort(key=lambda item: str(item["scope_key"]))
        return {
            "schema_version": LEADERBOARD_COMPARE_SNAPSHOT_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "group_count": len(groups_payload),
            "preferred_pair_count": len(preferred_pairs),
            "groups": groups_payload,
            "preferred_pairs": preferred_pairs,
        }

    @staticmethod
    def _merge_hardware_info(artifact: dict[str, Any]) -> dict[str, Any]:
        canonical = dict(artifact.get("hardware") or {})
        cluster = dict(artifact.get("cluster") or {})
        detected = (
            LeaderboardExporter.detect_hardware_info()
            if not {
                "vendor",
                "chip_model",
                "chip_count",
            }.issubset(canonical.keys())
            else {}
        )

        node_count = int(cluster.get("node_count") or canonical.get("node_count") or 1)
        chips_per_node = int(
            canonical.get("chips_per_node")
            or canonical.get("chip_count")
            or detected.get("chips_per_node")
            or detected.get("chip_count")
            or 1
        )
        chip_count = int(canonical.get("chip_count") or chips_per_node * node_count)

        return {
            "vendor": LeaderboardExporter._normalize_vendor(
                canonical.get("vendor") or detected.get("vendor")
            ),
            "chip_model": str(
                canonical.get("chip_model") or detected.get("chip_model") or "unknown"
            ),
            "chip_count": chip_count,
            "chips_per_node": int(canonical.get("chips_per_node") or chips_per_node),
            "interconnect": str(
                canonical.get("interconnect") or detected.get("interconnect") or "None"
            ),
            "intra_node_interconnect": str(
                canonical.get("intra_node_interconnect")
                or detected.get("intra_node_interconnect")
                or canonical.get("interconnect")
                or "None"
            ),
            "memory_per_chip_gb": canonical.get("memory_per_chip_gb")
            if canonical.get("memory_per_chip_gb") is not None
            else detected.get("memory_per_chip_gb"),
            "total_memory_gb": canonical.get("total_memory_gb")
            if canonical.get("total_memory_gb") is not None
            else detected.get("total_memory_gb"),
            "family": str(canonical.get("family") or detected.get("family") or "unknown"),
            "node_count": node_count,
        }

    @staticmethod
    def _build_cluster_info(
        artifact: dict[str, Any], hardware: dict[str, Any]
    ) -> dict[str, Any] | None:
        cluster = dict(artifact.get("cluster") or {})
        node_count = int(cluster.get("node_count") or hardware.get("node_count") or 1)
        if node_count <= 1:
            return None

        family = str(hardware.get("family") or "").lower()
        default_comm = {"ascend": "hccl", "cuda": "nccl", "rocm": "rccl"}.get(family, "gloo")
        return {
            "node_count": node_count,
            "comm_backend": str(cluster.get("comm_backend") or default_comm),
            "inter_node_network": str(
                cluster.get("inter_node_network") or hardware.get("interconnect") or "Ethernet"
            ),
            "network_bandwidth_gbps": cluster.get("network_bandwidth_gbps"),
            "topology_type": str(cluster.get("topology_type") or "multi_node"),
            "parallelism": cluster.get("parallelism"),
        }

    @staticmethod
    def _build_model_info_from_canonical(artifact: dict[str, Any]) -> dict[str, Any]:
        model = dict(artifact.get("model") or {})
        engine = dict(artifact.get("engine") or {})
        workload = dict(artifact.get("workload") or {})
        measurements = dict(artifact.get("measurements") or {})
        rows = measurements.get("rows") if isinstance(measurements.get("rows"), list) else []
        row_precision = rows[0].get("precision") if rows and isinstance(rows[0], dict) else None
        producer = dict(artifact.get("producer") or {})
        default_precision = (
            "FP16" if producer.get("command") in {"compare", "vllm-compare"} else "FP32"
        )

        return {
            "name": str(model.get("name") or engine.get("model") or "unknown"),
            "parameters": str(model.get("parameters") or "unknown"),
            "precision": LeaderboardExporter._normalize_precision(
                model.get("precision")
                or engine.get("precision")
                or workload.get("precision")
                or row_precision,
                default=default_precision,
            ),
            "quantization": model.get("quantization") or engine.get("quantization") or "None",
        }

    @staticmethod
    def _build_workload_info_from_canonical(artifact: dict[str, Any]) -> dict[str, Any]:
        workload = dict(artifact.get("workload") or {})
        measurements = dict(artifact.get("measurements") or {})
        rows = measurements.get("rows") if isinstance(measurements.get("rows"), list) else []
        workload_name = str(
            workload.get("name")
            or workload.get("workload_id")
            or workload.get("suite_id")
            or "LEGACY"
        )
        spec = LeaderboardExporter._extract_workload_spec(workload_name)

        batch_values = sorted(
            {
                int(row.get("batch_size"))
                for row in rows
                if isinstance(row, dict) and row.get("batch_size") is not None
            }
        )
        batch_size = workload.get("batch_size")
        if batch_size is None and len(batch_values) == 1:
            batch_size = batch_values[0]

        concurrency = workload.get("concurrency") or workload.get("concurrent_requests")
        if concurrency is None and workload.get("mode") == "live-compare":
            concurrency = batch_size

        return {
            "name": workload_name,
            "input_length": int(
                workload.get("input_length")
                or workload.get("prompt_tokens")
                or spec["input_length"]
            ),
            "output_length": int(
                workload.get("output_length") or workload.get("max_tokens") or spec["output_length"]
            ),
            "batch_size": int(batch_size) if batch_size is not None else spec.get("batch_size"),
            "concurrent_requests": int(concurrency)
            if concurrency is not None
            else spec.get("concurrent_requests"),
            "dataset": workload.get("dataset"),
            "suite_id": workload.get("suite_id"),
            "scenario_id": workload.get("scenario_id") or workload.get("mode"),
            "hardware_family": artifact.get("hardware", {}).get("family"),
        }

    @staticmethod
    def _build_metrics_from_canonical(artifact: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(artifact.get("metrics") or {})
        return {
            "ttft_ms": LeaderboardExporter._require_number(
                metrics.get("ttft_ms"), field_name="leaderboard entry.metrics.ttft_ms"
            ),
            "tbt_ms": LeaderboardExporter._require_number(
                metrics.get("tbt_ms"), field_name="leaderboard entry.metrics.tbt_ms"
            ),
            "tpot_ms": LeaderboardExporter._require_number(
                metrics.get("tpot_ms"), field_name="leaderboard entry.metrics.tpot_ms"
            ),
            "itl_ms": LeaderboardExporter._require_number(
                metrics.get("itl_ms"), field_name="leaderboard entry.metrics.itl_ms"
            ),
            "p50_itl_ms": LeaderboardExporter._coerce_optional_float(metrics.get("p50_itl_ms")),
            "p95_itl_ms": LeaderboardExporter._coerce_optional_float(metrics.get("p95_itl_ms")),
            "p99_itl_ms": LeaderboardExporter._coerce_optional_float(metrics.get("p99_itl_ms")),
            "e2el_ms": LeaderboardExporter._require_number(
                metrics.get("e2el_ms"), field_name="leaderboard entry.metrics.e2el_ms"
            ),
            "p50_e2el_ms": LeaderboardExporter._coerce_optional_float(metrics.get("p50_e2el_ms")),
            "p95_e2el_ms": LeaderboardExporter._coerce_optional_float(metrics.get("p95_e2el_ms")),
            "p99_e2el_ms": LeaderboardExporter._coerce_optional_float(metrics.get("p99_e2el_ms")),
            "throughput_tps": LeaderboardExporter._require_number(
                metrics.get("throughput_tps"), field_name="leaderboard entry.metrics.throughput_tps"
            ),
            "request_throughput_rps": LeaderboardExporter._require_number(
                metrics.get("request_throughput_rps"),
                field_name="leaderboard entry.metrics.request_throughput_rps",
            ),
            "input_throughput_tps": LeaderboardExporter._coerce_optional_float(
                metrics.get("input_throughput_tps")
            ),
            "output_throughput_tps": LeaderboardExporter._require_number(
                metrics.get("output_throughput_tps"),
                field_name="leaderboard entry.metrics.output_throughput_tps",
            ),
            "total_throughput_tps": LeaderboardExporter._coerce_optional_float(
                metrics.get("total_throughput_tps")
            ),
            "total_input_tokens": LeaderboardExporter._coerce_optional_int(
                metrics.get("total_input_tokens")
            ),
            "total_output_tokens": LeaderboardExporter._coerce_optional_int(
                metrics.get("total_output_tokens")
            ),
            "peak_mem_mb": int(metrics.get("peak_mem_mb") or 0),
            "error_rate": float(metrics.get("error_rate") or 0.0),
            "prefix_hit_rate": float(metrics.get("prefix_hit_rate") or 0.0),
            "kv_used_tokens": int(metrics.get("kv_used_tokens") or 0),
            "kv_used_bytes": int(metrics.get("kv_used_bytes") or 0),
            "evict_count": int(metrics.get("evict_count") or 0),
            "evict_ms": float(metrics.get("evict_ms") or 0.0),
            "spec_accept_rate": LeaderboardExporter._coerce_optional_float(
                metrics.get("spec_accept_rate")
            ),
        }

    @staticmethod
    def _validate_stream_compare_export_source(artifact: dict[str, Any]) -> None:
        schema_version = LeaderboardExporter._require_text(
            artifact.get("schema_version"), field_name="canonical artifact.schema_version"
        )
        if schema_version != SUPPORTED_CANONICAL_EXPORT_SCHEMA_VERSION:
            raise ValueError(
                "leaderboard export requires canonical compare artifacts with "
                f"schema_version {SUPPORTED_CANONICAL_EXPORT_SCHEMA_VERSION!r}"
            )

        if artifact.get("artifact_kind") != "execution_result":
            raise ValueError(
                "leaderboard export only supports canonical execution_result artifacts"
            )

        producer = LeaderboardExporter._require_mapping(
            artifact.get("producer"), field_name="canonical artifact.producer"
        )
        if producer.get("command") != "compare":
            raise ValueError(
                "leaderboard export only supports canonical live compare artifacts; "
                f"got producer.command={producer.get('command')!r}"
            )

        workload = LeaderboardExporter._require_mapping(
            artifact.get("workload"), field_name="canonical artifact.workload"
        )
        if workload.get("mode") != "live-compare":
            raise ValueError(
                "leaderboard export only supports canonical live compare artifacts; "
                f"got workload.mode={workload.get('mode')!r}"
            )

    @staticmethod
    def _build_versions_from_payload(versions: dict[str, Any]) -> tuple[dict[str, Any], str]:
        component_versions = {
            "protocol": LeaderboardExporter._resolve_version(
                versions, ("sagellm_protocol", "protocol")
            ),
            "backend": LeaderboardExporter._resolve_version(
                versions, ("sagellm_backend", "backend")
            ),
            "core": LeaderboardExporter._resolve_version(versions, ("sagellm_core", "core")),
            "control_plane": LeaderboardExporter._resolve_version(
                versions, ("sagellm_control_plane", "control_plane")
            ),
            "gateway": LeaderboardExporter._resolve_version(
                versions, ("sagellm_gateway", "gateway")
            ),
            "kv_cache": LeaderboardExporter._resolve_version(
                versions, ("sagellm_kv_cache", "kv_cache")
            ),
            "comm": LeaderboardExporter._resolve_version(versions, ("sagellm_comm", "comm")),
            "compression": LeaderboardExporter._resolve_version(
                versions, ("sagellm_compression", "compression")
            ),
            "benchmark": LeaderboardExporter._resolve_version(
                versions, ("sagellm_benchmark", "benchmark")
            ),
        }
        sagellm_version = LeaderboardExporter._resolve_version(
            versions,
            ("sagellm", "sagellm_benchmark", "benchmark"),
            default=component_versions["benchmark"],
        )
        return component_versions, sagellm_version

    @staticmethod
    def leaderboard_entry_from_canonical_artifact(
        artifact: dict[str, Any],
        custom_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        LeaderboardExporter._validate_stream_compare_export_source(artifact)

        producer = dict(artifact.get("producer") or {})
        provenance = dict(artifact.get("provenance") or {})
        engine_payload = dict(artifact.get("engine") or {})
        validation = dict(artifact.get("validation") or {})
        versions_payload = dict(artifact.get("versions") or {})

        hardware = LeaderboardExporter._merge_hardware_info(artifact)
        cluster = LeaderboardExporter._build_cluster_info(artifact, hardware)
        config_type = LeaderboardExporter.infer_config_type(
            int(hardware.get("chips_per_node") or hardware.get("chip_count") or 1),
            has_cluster=cluster is not None,
        )
        model_info = LeaderboardExporter._build_model_info_from_canonical(artifact)
        workload_info = LeaderboardExporter._build_workload_info_from_canonical(artifact)
        leaderboard_metrics = LeaderboardExporter._build_metrics_from_canonical(artifact)
        component_versions, sagellm_version = LeaderboardExporter._build_versions_from_payload(
            versions_payload
        )
        environment = dict(artifact.get("environment") or {})
        if not environment:
            environment = LeaderboardExporter.detect_environment()

        engine = LeaderboardExporter._normalize_engine_name(engine_payload.get("name"))
        engine_version = str(
            engine_payload.get("version")
            or LeaderboardExporter._resolve_version(
                versions_payload,
                (
                    engine.replace("-", "_"),
                    "vllm_ascend",
                    "vllm",
                    "lmdeploy",
                    "sagellm",
                    "benchmark",
                ),
            )
        )

        cmd_model = model_info["name"]
        cmd_workload = workload_info.get("name") or "unknown"
        if producer.get("command") == "run":
            reproducible_cmd = f"sagellm-benchmark run --workload {cmd_workload} --backend {engine_payload.get('backend', engine)} --model {cmd_model}"
        elif producer.get("command") == "compare":
            reproducible_cmd = f"sagellm-benchmark compare --target {engine}={provenance.get('endpoint_url', '')} --model {cmd_model} --hardware-family {hardware.get('family', 'unknown')}"
        else:
            reproducible_cmd = f"sagellm-benchmark {producer.get('command', 'run')}"

        metadata = {
            "submitted_at": str(provenance.get("captured_at") or datetime.now(UTC).isoformat()),
            "submitter": "sagellm-benchmark automated run",
            "data_source": str(producer.get("command") or "canonical-artifact"),
            "engine": engine,
            "engine_version": engine_version,
            "hardware_family": hardware.get("family"),
            "reproducible_cmd": reproducible_cmd,
            "git_commit": provenance.get("git", {}).get("commit")
            if isinstance(provenance.get("git"), dict)
            else None,
            "release_date": str(provenance.get("captured_at") or "").split("T")[0] or None,
            "changelog_url": "https://github.com/intellistream/sagellm/blob/main/CHANGELOG.md",
            "notes": f"Benchmark run: {cmd_workload}",
            "verified": bool(validation.get("publishable_to_leaderboard", False)),
            "canonical_artifact_id": artifact.get("artifact_id"),
            "canonical_artifact_kind": artifact.get("artifact_kind"),
            "canonical_output_dir": provenance.get("output_dir"),
        }
        if custom_metadata:
            metadata.update(custom_metadata)

        entry = {
            "schema_version": LEADERBOARD_ENTRY_SCHEMA_VERSION,
            "entry_id": str(uuid.uuid4()),
            "engine": engine,
            "engine_version": engine_version,
            "sagellm_version": sagellm_version,
            "config_type": config_type,
            "hardware": {
                "vendor": hardware["vendor"],
                "chip_model": hardware["chip_model"],
                "chip_count": hardware["chip_count"],
                "interconnect": hardware["interconnect"],
                "chips_per_node": hardware["chips_per_node"],
                "intra_node_interconnect": hardware["intra_node_interconnect"],
                "memory_per_chip_gb": hardware.get("memory_per_chip_gb"),
                "total_memory_gb": hardware.get("total_memory_gb"),
            },
            "model": model_info,
            "workload": workload_info,
            "metrics": leaderboard_metrics,
            "cluster": cluster,
            "versions": component_versions,
            "environment": environment,
            "kv_cache_config": {
                "enabled": True,
                "eviction_policy": "LRU",
                "budget_tokens": 8192,
                "prefix_cache_enabled": True,
            },
            "metadata": metadata,
        }
        return LeaderboardExporter.annotate_entry_identity(entry)

    @staticmethod
    def export_canonical_artifact(
        artifact: dict[str, Any],
        output_path: Path | str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = LeaderboardExporter.leaderboard_entry_from_canonical_artifact(
            artifact,
            custom_metadata=custom_metadata,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        return entry

    @staticmethod
    def export_to_leaderboard(
        metrics: AggregatedMetrics,
        config: dict[str, Any],
        workload_name: str,
        output_path: Path,
        custom_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise ValueError(
            "Direct leaderboard export from AggregatedMetrics has been removed; "
            "export only from canonical live compare artifacts"
        )

    @staticmethod
    def annotate_entry_identity(entry: dict[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(entry))
        LeaderboardExporter.validate_leaderboard_entry(normalized)
        metadata = normalized.setdefault("metadata", {})
        metadata["idempotency_key"] = LeaderboardExporter.build_idempotency_key(normalized)
        normalized["canonical_path"] = LeaderboardExporter.build_canonical_path(normalized)
        return normalized

    @staticmethod
    def register_exported_entry(
        *,
        output_dir: Path | str,
        entry: dict[str, Any],
        leaderboard_path: Path | str,
        canonical_artifact_path: Path | str,
    ) -> Path:
        output_path = Path(output_dir)
        manifest_path = output_path / "leaderboard_manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != LEADERBOARD_MANIFEST_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported leaderboard manifest schema: {payload.get('schema_version')!r}"
                )
            records = payload.get("entries")
            if not isinstance(records, list):
                raise ValueError("leaderboard_manifest.json entries must be a list")
        else:
            records = []

        normalized = LeaderboardExporter.annotate_entry_identity(entry)
        leaderboard_file = Path(leaderboard_path)
        canonical_file = Path(canonical_artifact_path)
        record = {
            "entry_id": normalized["entry_id"],
            "idempotency_key": normalized["metadata"]["idempotency_key"],
            "canonical_path": normalized["canonical_path"],
            "leaderboard_artifact": str(leaderboard_file.relative_to(output_path)),
            "canonical_artifact": str(canonical_file.relative_to(output_path)),
            "engine": normalized["engine"],
            "workload": normalized.get("workload", {}).get("name"),
            "config_type": normalized["config_type"],
            "category": LeaderboardExporter._entry_bucket(normalized),
        }

        by_key = {
            str(existing.get("idempotency_key")): existing
            for existing in records
            if isinstance(existing, dict) and existing.get("idempotency_key")
        }
        by_key[record["idempotency_key"]] = record
        manifest_payload = {
            "schema_version": LEADERBOARD_MANIFEST_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "entries": sorted(
                by_key.values(), key=lambda item: str(item.get("leaderboard_artifact"))
            ),
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
        return manifest_path

    @staticmethod
    def normalize_entries_payload(payload: dict | list) -> list[dict]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []

    @staticmethod
    def build_idempotency_key(entry: dict[str, Any]) -> str:
        cluster = entry.get("cluster") or {}

        def normalize(value: Any) -> str:
            text = str(value if value is not None else "unknown").strip().lower()
            return re.sub(r"[^a-z0-9._-]+", "-", text).strip("-") or "unknown"

        parts = [
            normalize(entry.get("engine") or entry.get("metadata", {}).get("engine") or "unknown"),
            normalize(
                entry.get("engine_version")
                or entry.get("metadata", {}).get("engine_version")
                or entry.get("sagellm_version")
                or "unknown"
            ),
            normalize(entry.get("sagellm_version")),
            normalize(LeaderboardExporter._extract_workload_name_from_entry(entry)),
            normalize(entry.get("model", {}).get("name")),
            normalize(entry.get("model", {}).get("precision")),
            normalize(entry.get("hardware", {}).get("chip_model")),
            normalize(entry.get("hardware", {}).get("chip_count")),
            normalize(cluster.get("node_count", 1)),
            normalize(entry.get("config_type")),
        ]
        return "|".join(parts)

    @staticmethod
    def build_canonical_path(entry: dict[str, Any]) -> str:
        digest = hashlib.sha1(
            LeaderboardExporter.build_idempotency_key(entry).encode("utf-8")
        ).hexdigest()[:20]
        return f"canonical/{digest}_leaderboard.json"

    @staticmethod
    def parse_entry_time(entry: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
        metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        submitted_raw = metadata.get("submitted_at")
        release_raw = metadata.get("release_date")

        submitted_dt = None
        if isinstance(submitted_raw, str) and submitted_raw:
            try:
                submitted_dt = datetime.fromisoformat(submitted_raw.replace("Z", "+00:00"))
            except ValueError:
                submitted_dt = None

        release_dt = None
        if isinstance(release_raw, str) and release_raw:
            try:
                release_dt = datetime.fromisoformat(release_raw)
            except ValueError:
                release_dt = None

        return submitted_dt, release_dt

    @staticmethod
    def prefer_newer_entry(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        current_submitted, current_release = LeaderboardExporter.parse_entry_time(current)
        candidate_submitted, candidate_release = LeaderboardExporter.parse_entry_time(candidate)

        if current_submitted and candidate_submitted and candidate_submitted != current_submitted:
            return candidate if candidate_submitted > current_submitted else current
        if current_submitted is None and candidate_submitted is not None:
            return candidate
        if current_submitted is not None and candidate_submitted is None:
            return current

        if current_release and candidate_release and candidate_release != current_release:
            return candidate if candidate_release > current_release else current
        if current_release is None and candidate_release is not None:
            return candidate
        if current_release is not None and candidate_release is None:
            return current

        current_tps = float(current.get("metrics", {}).get("throughput_tps") or 0.0)
        candidate_tps = float(candidate.get("metrics", {}).get("throughput_tps") or 0.0)
        if candidate_tps != current_tps:
            return candidate if candidate_tps > current_tps else current

        return current

    @staticmethod
    def collect_entries_from_directory(
        input_dir: Path | str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        input_path = Path(input_dir)
        parse_errors: list[str] = []
        entries: list[dict[str, Any]] = []

        manifest_files = sorted(input_path.rglob("leaderboard_manifest.json"))
        if not manifest_files:
            parse_errors.append(f"No leaderboard_manifest.json found under: {input_path}")
            return entries, parse_errors

        for manifest_path in manifest_files:
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                parse_errors.append(f"{manifest_path}: {exc}")
                continue

            if manifest_payload.get("schema_version") != LEADERBOARD_MANIFEST_SCHEMA_VERSION:
                parse_errors.append(
                    f"{manifest_path}: unsupported schema_version {manifest_payload.get('schema_version')!r}"
                )
                continue

            records = manifest_payload.get("entries")
            if not isinstance(records, list):
                parse_errors.append(f"{manifest_path}: entries must be a list")
                continue

            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    parse_errors.append(f"{manifest_path}: entries[{index}] must be an object")
                    continue

                try:
                    artifact_rel = LeaderboardExporter._require_text(
                        record.get("leaderboard_artifact"),
                        field_name=f"{manifest_path}.entries[{index}].leaderboard_artifact",
                    )
                    expected_idempotency = LeaderboardExporter._require_text(
                        record.get("idempotency_key"),
                        field_name=f"{manifest_path}.entries[{index}].idempotency_key",
                    )
                    expected_canonical_path = LeaderboardExporter._require_text(
                        record.get("canonical_path"),
                        field_name=f"{manifest_path}.entries[{index}].canonical_path",
                    )
                except ValueError as exc:
                    parse_errors.append(str(exc))
                    continue

                artifact_path = manifest_path.parent / artifact_rel
                if not artifact_path.is_file():
                    parse_errors.append(
                        f"{manifest_path}: missing leaderboard artifact {artifact_path}"
                    )
                    continue

                try:
                    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    parse_errors.append(f"{artifact_path}: {exc}")
                    continue

                if not isinstance(payload, dict):
                    parse_errors.append(
                        f"{artifact_path}: standard exported leaderboard artifact must be a JSON object"
                    )
                    continue

                try:
                    entry = LeaderboardExporter.annotate_entry_identity(payload)
                except Exception as exc:
                    parse_errors.append(f"{artifact_path}: {exc}")
                    continue

                if entry["metadata"]["idempotency_key"] != expected_idempotency:
                    parse_errors.append(
                        f"{artifact_path}: idempotency_key mismatch with manifest {manifest_path}"
                    )
                    continue
                if entry["canonical_path"] != expected_canonical_path:
                    parse_errors.append(
                        f"{artifact_path}: canonical_path mismatch with manifest {manifest_path}"
                    )
                    continue

                entry.setdefault("metadata", {})["manifest_source"] = str(manifest_path)
                entries.append(entry)

        return entries, parse_errors

    @staticmethod
    def build_snapshot_payloads(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        deduped: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(entries):
            normalized = LeaderboardExporter.annotate_entry_identity(entry)
            key = normalized["metadata"]["idempotency_key"]
            existing = deduped.get(key)
            deduped[key] = (
                LeaderboardExporter.prefer_newer_entry(existing, normalized)
                if existing is not None
                else normalized
            )

        single: list[dict[str, Any]] = []
        multi: list[dict[str, Any]] = []
        for entry in deduped.values():
            bucket = LeaderboardExporter._entry_bucket(entry)
            if bucket == "multi":
                multi.append(entry)
            else:
                single.append(entry)

        def sort_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
            return (
                str(item.get("engine") or ""),
                str(item.get("model", {}).get("name") or ""),
                str(item.get("workload", {}).get("name") or ""),
                str(item.get("metadata", {}).get("submitted_at") or ""),
            )

        single.sort(key=sort_key)
        multi.sort(key=sort_key)
        return {"single": single, "multi": multi}
