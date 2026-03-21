"""Performance regression detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetricCheck:
    """A metric comparison result against baseline."""

    name: str
    baseline: float
    current: float
    regression_pct: float
    better_is_higher: bool
    expected_change: bool = False

    @property
    def direction(self) -> str:
        return "higher-is-better" if self.better_is_higher else "lower-is-better"


def extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Extract normalized metric summary from benchmark payload."""
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return {
            "avg_ttft_ms": float(summary.get("avg_ttft_ms", 0.0)),
            "avg_tbt_ms": float(summary.get("avg_tbt_ms", 0.0)),
            "avg_throughput_tps": float(summary.get("avg_throughput_tps", 0.0)),
            "peak_mem_mb": float(summary.get("peak_mem_mb", 0.0)),
            "error_rate": float(summary.get("error_rate", 0.0)),
        }

    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Cannot extract metrics: payload has no summary or rows.")

    avg_ttft = sum(float(row.get("ttft_ms", 0.0)) for row in rows) / len(rows)
    avg_tbt = sum(float(row.get("tbt_ms", 0.0)) for row in rows) / len(rows)
    avg_tps = sum(float(row.get("throughput_tps", 0.0)) for row in rows) / len(rows)
    peak_mem = max(float(row.get("memory_mb", row.get("peak_mem_mb", 0.0)) or 0.0) for row in rows)
    error_rate = sum(1 for row in rows if row.get("ok") is False) / len(rows)
    return {
        "avg_ttft_ms": avg_ttft,
        "avg_tbt_ms": avg_tbt,
        "avg_throughput_tps": avg_tps,
        "peak_mem_mb": peak_mem,
        "error_rate": error_rate,
    }


class RegressionDetector:
    """Detect benchmark regressions using thresholds and expected changes."""

    def __init__(
        self,
        warning_threshold_pct: float = 5.0,
        critical_threshold_pct: float = 10.0,
        expected_changes: set[str] | None = None,
    ) -> None:
        self.warning_threshold_pct = warning_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct
        self.expected_changes = expected_changes or set()

    def compute_checks(
        self,
        baseline_metrics: dict[str, float],
        current_metrics: dict[str, float],
    ) -> list[MetricCheck]:
        """Compute metric checks from normalized summaries."""
        checks: list[MetricCheck] = []

        for metric in ["avg_ttft_ms", "avg_tbt_ms", "peak_mem_mb", "error_rate"]:
            base_value = baseline_metrics[metric]
            current_value = current_metrics[metric]
            regression = (
                (current_value - base_value) / base_value * 100.0 if base_value > 0 else 0.0
            )
            checks.append(
                MetricCheck(
                    name=metric,
                    baseline=base_value,
                    current=current_value,
                    regression_pct=regression,
                    better_is_higher=False,
                    expected_change=metric in self.expected_changes,
                )
            )

        metric = "avg_throughput_tps"
        base_value = baseline_metrics[metric]
        current_value = current_metrics[metric]
        regression = (base_value - current_value) / base_value * 100.0 if base_value > 0 else 0.0
        checks.append(
            MetricCheck(
                name=metric,
                baseline=base_value,
                current=current_value,
                regression_pct=regression,
                better_is_higher=True,
                expected_change=metric in self.expected_changes,
            )
        )
        return checks

    def classify_status(self, check: MetricCheck) -> str:
        """Classify check status."""
        if check.expected_change:
            return "expected-change"
        if check.regression_pct > self.critical_threshold_pct:
            return "critical"
        if check.regression_pct > self.warning_threshold_pct:
            return "warning"
        return "acceptable"

    def build_summary(self, checks: list[MetricCheck]) -> dict[str, Any]:
        """Build structured summary from checks."""
        metrics: dict[str, Any] = {}
        overall = "acceptable"

        for check in checks:
            status = self.classify_status(check)
            if status == "critical":
                overall = "critical"
            elif status == "warning" and overall != "critical":
                overall = "warning"

            metrics[check.name] = {
                "baseline": check.baseline,
                "current": check.current,
                "regression_pct": check.regression_pct,
                "status": status,
                "direction": check.direction,
                "expected_change": check.expected_change,
            }

        return {
            "overall_status": overall,
            "warning_threshold_pct": self.warning_threshold_pct,
            "critical_threshold_pct": self.critical_threshold_pct,
            "metrics": metrics,
        }

    def compare(
        self,
        baseline_payload: dict[str, Any],
        current_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare payloads and return summary."""
        baseline_metrics = extract_metrics(baseline_payload)
        current_metrics = extract_metrics(current_payload)
        checks = self.compute_checks(baseline_metrics, current_metrics)
        return self.build_summary(checks)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render markdown report from summary."""
    lines = [
        "# Performance Regression Report",
        "",
        f"- Overall Status: **{summary['overall_status'].upper()}**",
        f"- Warning Threshold: {summary['warning_threshold_pct']:.1f}%",
        f"- Critical Threshold: {summary['critical_threshold_pct']:.1f}%",
        "",
        "| Metric | Baseline | Current | Regression % | Status |",
        "|--------|----------|---------|--------------|--------|",
    ]

    for metric_name, metric in summary["metrics"].items():
        status_text = str(metric["status"])
        if metric.get("expected_change"):
            status_text = f"{status_text} (allowlisted)"
        lines.append(
            f"| {metric_name} | {float(metric['baseline']):.4f} | {float(metric['current']):.4f} | "
            f"{float(metric['regression_pct']):.2f}% | {status_text} |"
        )

    return "\n".join(lines)
