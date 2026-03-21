"""Baseline management utilities for benchmark regression checks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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
class BenchmarkBaselineConfig:
    enabled: bool
    kill_switch_active: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> BenchmarkBaselineConfig:
        source = env or dict(os.environ)
        enabled_raw = source.get("SAGELLM_BENCH_CR", "0")
        kill_raw = source.get("SAGELLM_BENCH_CR_KILL", "0")
        return cls(
            enabled=_parse_bool_env(enabled_raw, "SAGELLM_BENCH_CR"),
            kill_switch_active=_parse_bool_env(kill_raw, "SAGELLM_BENCH_CR_KILL"),
        )


@dataclass
class BaselineManager:
    """Manage persisted benchmark baselines."""

    baseline_path: Path

    def load(self) -> dict[str, Any]:
        """Load baseline payload from disk."""
        with self.baseline_path.open(encoding="utf-8") as file:
            return json.load(file)

    def save(self, payload: dict[str, Any]) -> None:
        """Persist baseline payload to disk."""
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with self.baseline_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Update baseline with current benchmark payload and metadata."""
        updated_payload = dict(payload)
        metadata = dict(updated_payload.get("metadata", {}))
        metadata["baseline_updated_at"] = datetime.now(UTC).isoformat()
        config = BenchmarkBaselineConfig.from_env()
        metadata.setdefault(
            "benchmark_contract",
            {
                "enabled": config.enabled,
                "kill_switch_active": config.kill_switch_active,
                "threshold_metrics": [
                    "avg_ttft_ms",
                    "avg_tbt_ms",
                    "avg_throughput_tps",
                    "peak_mem_mb",
                    "error_rate",
                ],
            },
        )
        updated_payload["metadata"] = metadata
        self.save(updated_payload)
        return updated_payload
