"""CLI for sagellm-benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sagellm_benchmark.core_telemetry import build_core_decode_telemetry_artifact
from sagellm_benchmark.nonstream_compare import (
    NonStreamCompareConfig,
    parse_target_spec,
    run_nonstream_compare,
)
from sagellm_benchmark.parity_gate import (
    DecodeParityGate,
    build_default_cuda_decode_gate,
    build_parity_run_artifact_from_e2e_payload,
    evaluate_parity_gate,
    load_parity_run_artifact,
)
from sagellm_benchmark.runtime_consistency import (
    build_live_runtime_consistency_report,
    extract_runtime_info_payload,
)

console = Console()


def _slugify_filename(value: str) -> str:
    """Convert a label to a filesystem-safe filename stem."""
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "target"


def _parse_compare_target(spec: str) -> tuple[str, str]:
    """Parse compare target spec in LABEL=URL format."""
    if "=" not in spec:
        raise click.BadParameter(
            f"Invalid target '{spec}'. Expected LABEL=URL, for example sagellm=http://127.0.0.1:8000/v1"
        )

    label, url = spec.split("=", 1)
    label = label.strip()
    url = url.strip()
    if not label or not url:
        raise click.BadParameter(
            f"Invalid target '{spec}'. Both label and URL are required in LABEL=URL format."
        )
    return label, url


def _parse_label_command(spec: str) -> tuple[str, str]:
    """Parse a LABEL=COMMAND mapping."""
    if "=" not in spec:
        raise click.BadParameter(
            f"Invalid command mapping '{spec}'. Expected LABEL=COMMAND, for example sagellm='sagellm serve --port 8901'"
        )

    label, command = spec.split("=", 1)
    label = label.strip()
    command = command.strip()
    if not label or not command:
        raise click.BadParameter(
            f"Invalid command mapping '{spec}'. Both label and command are required in LABEL=COMMAND format."
        )
    return label, command


def _parse_label_path(spec: str) -> tuple[str, str]:
    """Parse a LABEL=PATH mapping."""
    if "=" not in spec:
        raise click.BadParameter(
            f"Invalid result mapping '{spec}'. Expected LABEL=PATH, for example sagellm=./results/sagellm.json"
        )

    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise click.BadParameter(
            f"Invalid result mapping '{spec}'. Both label and path are required in LABEL=PATH format."
        )
    return label, path


def _build_compare_summary(target_results: list[dict[str, object]]) -> dict[str, object]:
    """Build a comparison summary using the first target as baseline."""
    if not target_results:
        return {"baseline": None, "targets": []}

    baseline = target_results[0]
    baseline_summary = baseline["summary"]
    baseline_ttft = float(baseline_summary["avg_ttft_ms"])
    baseline_tbt = float(baseline_summary["avg_tbt_ms"])
    baseline_tps = float(baseline_summary["avg_throughput_tps"])

    summary_rows: list[dict[str, object]] = []
    for target in target_results:
        target_summary = target["summary"]
        summary_rows.append(
            {
                "label": target["label"],
                "url": target["url"],
                "rows": int(target_summary["total_rows"]),
                "avg_ttft_ms": float(target_summary["avg_ttft_ms"]),
                "avg_tbt_ms": float(target_summary["avg_tbt_ms"]),
                "avg_throughput_tps": float(target_summary["avg_throughput_tps"]),
                "delta_vs_baseline": {
                    "ttft_ms": float(target_summary["avg_ttft_ms"]) - baseline_ttft,
                    "tbt_ms": float(target_summary["avg_tbt_ms"]) - baseline_tbt,
                    "throughput_tps": float(target_summary["avg_throughput_tps"]) - baseline_tps,
                },
            }
        )

    return {
        "baseline": baseline["label"],
        "targets": summary_rows,
    }


def _format_compare_markdown(compare_result: dict[str, object]) -> str:
    """Render markdown summary for a compare run."""
    lines = [
        "# Endpoint Comparison Report",
        "",
        f"- Model: {compare_result['model']}",
        f"- Baseline: {compare_result['baseline']}",
        f"- Batch sizes: {', '.join(str(size) for size in compare_result['batch_sizes'])}",
        "",
        "| Target | Rows | Avg TTFT (ms) | Avg TBT (ms) | Avg TPS | Delta TTFT | Delta TBT | Delta TPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for target in compare_result["targets"]:
        delta = target["delta_vs_baseline"]
        lines.append(
            "| {label} | {rows} | {ttft:.2f} | {tbt:.2f} | {tps:.2f} | {d_ttft:+.2f} | {d_tbt:+.2f} | {d_tps:+.2f} |".format(
                label=target["label"],
                rows=target["rows"],
                ttft=float(target["avg_ttft_ms"]),
                tbt=float(target["avg_tbt_ms"]),
                tps=float(target["avg_throughput_tps"]),
                d_ttft=float(delta["ttft_ms"]),
                d_tbt=float(delta["tbt_ms"]),
                d_tps=float(delta["throughput_tps"]),
            )
        )

    lines.extend(["", "## Targets", ""])
    for target in compare_result["targets"]:
        lines.append(f"- {target['label']}: {target['url']}")

    return "\n".join(lines)


def _run_compare_target(
    *,
    label: str,
    url: str,
    model: str,
    hardware_family: str,
    batch_sizes: tuple[int, ...],
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
    max_seq_len: int | None,
    max_output_tokens: int | None,
    output_dir: Path,
) -> dict[str, object]:
    """Run a single live compare target and write per-target artifacts."""
    from sagellm_benchmark.performance.model_benchmarks import (
        run_e2e_model_benchmarks,
        summarize_e2e_rows,
    )

    rows = run_e2e_model_benchmarks(
        models=[model],
        batch_sizes=list(batch_sizes),
        precisions=["live"],
        simulate=False,
        backend_url=url,
        api_key=api_key,
        request_timeout=request_timeout,
        server_wait_s=server_wait_s,
        max_seq_len=max_seq_len,
        max_output_tokens=max_output_tokens,
    )
    summary = summarize_e2e_rows(rows)
    runtime_artifacts = _capture_target_runtime_artifacts(
        label=label,
        url=url,
        model=model,
        hardware_family=hardware_family,
        api_key=api_key,
        request_timeout=request_timeout,
        output_dir=output_dir,
    )
    _validate_sagellm_explicit_decode_runtime(
        label=label,
        runtime_artifacts=runtime_artifacts,
    )
    payload = {
        "kind": "e2e",
        "simulate": False,
        "mode": "live-compare",
        "label": label,
        "url": url,
        "hardware_family": hardware_family,
        "models": [model],
        "batch_sizes": list(batch_sizes),
        "precisions": ["live"],
        "runtime_artifacts": runtime_artifacts,
        "summary": summary,
        "rows": rows,
    }
    parity_artifact = build_parity_run_artifact_from_e2e_payload(
        payload,
        hardware_family=hardware_family,
    )

    file_stem = _slugify_filename(label)
    json_path = output_dir / f"{file_stem}.json"
    md_path = output_dir / f"{file_stem}.md"
    parity_path = output_dir / f"{file_stem}.parity.json"

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(md_path, "w") as f:
        f.write(_format_e2e_markdown(payload) + "\n")
    parity_path.write_text(parity_artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")

    return {
        "label": label,
        "url": url,
        "summary": summary,
        "json": str(json_path),
        "markdown": str(md_path),
        "parity_json": str(parity_path),
        "runtime_artifacts": runtime_artifacts,
        "payload": payload,
    }


def _write_compare_summary_artifacts(
    *,
    compare_output_dir: Path,
    model: str,
    batch_sizes: list[int],
    target_results: list[dict[str, object]],
) -> dict[str, object]:
    """Write comparison summary artifacts from precomputed target results."""
    compare_result = {
        "kind": "compare",
        "model": model,
        "batch_sizes": batch_sizes,
        **_build_compare_summary(target_results),
    }

    comparison_json = compare_output_dir / "comparison.json"
    comparison_md = compare_output_dir / "comparison.md"
    with open(comparison_json, "w") as f:
        json.dump(compare_result, f, indent=2)
    with open(comparison_md, "w") as f:
        f.write(_format_compare_markdown(compare_result) + "\n")

    return compare_result


def _load_compare_result_payload(label: str, path: str) -> dict[str, object]:
    """Load a single-target compare result payload for offline comparison."""
    result_path = Path(path)
    if not result_path.is_file():
        raise click.ClickException(f"Result file not found: {result_path}")

    with open(result_path) as f:
        payload = json.load(f)

    if payload.get("kind") != "e2e":
        raise click.ClickException(
            f"Result file must contain an e2e payload produced by compare-record or compare: {result_path}"
        )
    if "summary" not in payload:
        raise click.ClickException(f"Result file is missing summary: {result_path}")

    file_label = str(payload.get("label") or label)
    if not file_label:
        raise click.ClickException(
            f"Result file is missing label and none was provided in LABEL=PATH: {result_path}"
        )

    payload["label"] = file_label
    payload["url"] = str(payload.get("url") or "offline-captured")
    return payload


def _create_compare_output_dir(output_dir: str | None, prefix: str = "compare") -> Path:
    """Create the output directory for compare-style commands."""
    return (
        Path(output_dir)
        if output_dir
        else Path("benchmark_results") / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def _is_local_target_url(url: str) -> bool:
    """Return whether a compare target points at a local endpoint."""
    parsed = urlparse(url)
    return parsed.hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _root_url_from_api_base(url: str) -> str:
    """Return the endpoint root URL for auxiliary probes like /info."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return parsed._replace(path=path or "/", params="", query="", fragment="").geturl().rstrip("/")


def _fetch_json_probe(
    url: str,
    *,
    api_key: str,
    timeout_s: float,
) -> dict[str, object] | None:
    """Fetch a JSON probe endpoint and return the decoded payload when available."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout_s) as response:
            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type.lower():
                return None
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else None
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def _capture_target_runtime_artifacts(
    *,
    label: str,
    url: str,
    model: str,
    hardware_family: str,
    api_key: str,
    request_timeout: float,
    output_dir: Path,
) -> dict[str, str]:
    """Best-effort capture of runtime metadata artifacts for compare targets."""
    info_payload = _fetch_json_probe(
        f"{_root_url_from_api_base(url)}/info",
        api_key=api_key,
        timeout_s=min(request_timeout, 5.0),
    )
    if info_payload is None:
        return {}

    file_stem = _slugify_filename(label)
    runtime_artifacts: dict[str, str] = {}

    info_path = output_dir / f"{file_stem}_info.json"
    info_path.write_text(json.dumps(info_payload, indent=2) + "\n", encoding="utf-8")
    runtime_artifacts["info_json"] = str(info_path)

    try:
        telemetry_source_payload = extract_runtime_info_payload(info_payload)
    except ValueError:
        return runtime_artifacts

    performance_mainline = telemetry_source_payload.get("performance_mainline")
    if not isinstance(performance_mainline, dict) or "explicit_decode" not in performance_mainline:
        return runtime_artifacts

    artifact = build_core_decode_telemetry_artifact(
        telemetry_source_payload,
        label=label,
        model=model,
        hardware_family=hardware_family,
    )
    telemetry_path = output_dir / f"{file_stem}_core_telemetry.json"
    telemetry_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    runtime_artifacts["core_telemetry_json"] = str(telemetry_path)
    return runtime_artifacts


def _validate_sagellm_explicit_decode_runtime(
    *,
    label: str,
    runtime_artifacts: dict[str, str],
) -> None:
    """Fail fast when a sagellm compare target is not running native explicit decode."""
    if not label.lower().startswith("sagellm"):
        return

    info_json_path = runtime_artifacts.get("info_json")
    if not info_json_path:
        raise click.ClickException(
            f"Target '{label}' did not expose /info runtime metadata; cannot verify native explicit decode."
        )

    info_payload = json.loads(Path(info_json_path).read_text(encoding="utf-8"))
    runtime_info = extract_runtime_info_payload(info_payload)
    performance_mainline = runtime_info.get("performance_mainline")
    if not isinstance(performance_mainline, dict):
        raise click.ClickException(
            f"Target '{label}' /info payload is missing performance_mainline; refusing to benchmark an unverified runtime path."
        )

    explicit_decode = performance_mainline.get("explicit_decode")
    if not isinstance(explicit_decode, dict):
        raise click.ClickException(
            f"Target '{label}' /info payload is missing performance_mainline.explicit_decode; refusing to benchmark an unverified runtime path."
        )

    feature_gate = explicit_decode.get("feature_gate")
    if not isinstance(feature_gate, dict):
        raise click.ClickException(
            f"Target '{label}' /info payload is missing explicit decode feature gate state; refusing to benchmark an unverified runtime path."
        )

    if not bool(feature_gate.get("default_enabled", False)):
        raise click.ClickException(
            f"Target '{label}' reports explicit decode default_enabled=false; fix the serving defaults before benchmarking."
        )
    if not bool(feature_gate.get("enabled", False)):
        raise click.ClickException(
            f"Target '{label}' reports explicit decode enabled=false; use the native explicit decode path before benchmarking."
        )
    if bool(feature_gate.get("kill_switch_active", False)):
        raise click.ClickException(
            f"Target '{label}' reports explicit decode kill_switch_active=true; benchmark aborted."
        )


def _discover_local_target_processes(
    parsed_targets: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Find listening local processes for compare target ports."""
    listeners = subprocess.run(
        ["ss", "-ltnpH"],
        capture_output=True,
        text=True,
        check=True,
    )

    by_pid: dict[int, dict[str, object]] = {}
    for label, url in parsed_targets:
        if not _is_local_target_url(url):
            continue

        parsed = urlparse(url)
        if parsed.port is None:
            continue

        for line in listeners.stdout.splitlines():
            if not re.search(rf":{parsed.port}\b", line):
                continue

            for pid_text in re.findall(r"pid=(\d+)", line):
                pid = int(pid_text)
                entry = by_pid.setdefault(
                    pid,
                    {
                        "pid": pid,
                        "labels": set(),
                        "urls": set(),
                        "ports": set(),
                    },
                )
                entry["labels"].add(label)
                entry["urls"].add(url)
                entry["ports"].add(parsed.port)

    for entry in by_pid.values():
        try:
            proc = subprocess.run(
                ["ps", "-p", str(entry["pid"]), "-o", "command="],
                capture_output=True,
                text=True,
                check=True,
            )
            entry["command"] = proc.stdout.strip()
        except subprocess.CalledProcessError:
            entry["command"] = ""

        entry["labels"] = sorted(entry["labels"])
        entry["urls"] = sorted(entry["urls"])
        entry["ports"] = sorted(entry["ports"])

    return sorted(by_pid.values(), key=lambda item: int(item["pid"]))


def _endpoint_is_ready(
    url: str,
    *,
    api_key: str,
    request_timeout: float,
    probe_timeout: float = 5.0,
) -> bool:
    """Check whether an OpenAI-compatible endpoint is ready."""
    from sagellm_benchmark.clients.openai_client import GatewayClient

    client = GatewayClient(base_url=url, api_key=api_key, timeout=request_timeout)
    return asyncio.run(client.health_check(timeout=probe_timeout))


def _wait_for_endpoint_ready(
    url: str,
    *,
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
) -> bool:
    """Wait for an endpoint to become ready within the given timeout."""
    deadline = time.time() + server_wait_s
    while time.time() < deadline:
        if _endpoint_is_ready(
            url,
            api_key=api_key,
            request_timeout=request_timeout,
        ):
            return True
        remaining = deadline - time.time()
        if remaining > 0:
            time.sleep(min(2.0, remaining))
    return False


def _process_is_alive(pid: int) -> bool:
    """Check whether a process is still alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_processes(
    pids: list[int],
    *,
    grace_period_s: float = 3.0,
) -> dict[str, list[int]]:
    """Terminate processes with TERM first, then KILL if needed."""
    terminated: list[int] = []
    killed: list[int] = []
    failed: list[int] = []

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            terminated.append(pid)
        except OSError:
            failed.append(pid)

    deadline = time.time() + grace_period_s
    while time.time() < deadline:
        remaining = [pid for pid in pids if _process_is_alive(pid)]
        if not remaining:
            break
        time.sleep(0.1)

    for pid in pids:
        if pid in failed:
            continue
        if not _process_is_alive(pid):
            if pid not in terminated:
                terminated.append(pid)
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            if pid not in terminated:
                terminated.append(pid)
        except OSError:
            failed.append(pid)

    return {
        "terminated": sorted(set(terminated)),
        "killed": sorted(set(killed)),
        "failed": sorted(set(failed)),
    }


def _terminate_process_groups(
    pgids: list[int],
    *,
    grace_period_s: float = 3.0,
) -> dict[str, list[int]]:
    """Terminate managed process groups started by benchmark compare."""
    terminated: list[int] = []
    killed: list[int] = []
    failed: list[int] = []

    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            terminated.append(pgid)
        except OSError:
            failed.append(pgid)

    deadline = time.time() + grace_period_s
    while time.time() < deadline:
        remaining = [pgid for pgid in pgids if _process_is_alive(pgid)]
        if not remaining:
            break
        time.sleep(0.1)

    for pgid in pgids:
        if pgid in failed:
            continue
        if not _process_is_alive(pgid):
            if pgid not in terminated:
                terminated.append(pgid)
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
            killed.append(pgid)
        except ProcessLookupError:
            if pgid not in terminated:
                terminated.append(pgid)
        except OSError:
            failed.append(pgid)

    return {
        "terminated": sorted(set(terminated)),
        "killed": sorted(set(killed)),
        "failed": sorted(set(failed)),
    }


def _maybe_start_local_targets(
    *,
    parsed_targets: list[tuple[str, str]],
    target_commands: dict[str, str],
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
) -> list[dict[str, object]]:
    """Start local targets when commands are provided and endpoints are not ready."""
    managed_processes: list[dict[str, object]] = []

    for label, url in parsed_targets:
        command = target_commands.get(label)
        if not command:
            continue
        if not _is_local_target_url(url):
            raise click.ClickException(
                f"Auto-start is only supported for local targets. '{label}' points to {url}."
            )

        if _endpoint_is_ready(url, api_key=api_key, request_timeout=request_timeout):
            console.print(f"[dim]{label} already running at {url}; skipping start command.[/dim]")
            continue

        if not command.strip():
            raise click.ClickException(f"Start command for target '{label}' is empty.")

        console.print(f"[bold]Starting target:[/bold] {label} -> {command}")
        process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            start_new_session=True,
        )
        managed_processes.append(
            {
                "pid": process.pid,
                "pgid": process.pid,
                "label": label,
                "url": url,
                "command": command,
                "started_by_benchmark": True,
            }
        )

        if not _wait_for_endpoint_ready(
            url,
            api_key=api_key,
            request_timeout=request_timeout,
            server_wait_s=server_wait_s,
        ):
            _terminate_process_groups([int(item["pgid"]) for item in managed_processes])
            raise click.ClickException(
                f"Started target '{label}' but endpoint {url} did not become ready within {server_wait_s:.0f}s."
            )

        console.print(f"[green]✓[/green] {label} became ready at {url}")

    return managed_processes


def _should_prompt_cleanup(prompt_cleanup: bool | None) -> bool:
    """Return whether compare should prompt to clean up local endpoints."""
    if prompt_cleanup is not None:
        return prompt_cleanup
    return sys.stdin.isatty() and sys.stdout.isatty()


def _maybe_prompt_cleanup_local_targets(
    parsed_targets: list[tuple[str, str]],
    *,
    prompt_cleanup: bool | None,
    managed_processes: list[dict[str, object]] | None = None,
) -> None:
    """Ask whether to terminate local target processes after compare completes."""
    if not _should_prompt_cleanup(prompt_cleanup):
        return

    local_processes = _discover_local_target_processes(parsed_targets)
    managed_processes = managed_processes or []
    managed_by_pid = {
        int(process["pid"]): {
            "pid": int(process["pid"]),
            "labels": [str(process["label"])],
            "urls": [str(process["url"])],
            "ports": [],
            "command": str(process["command"]),
            "pgid": int(process["pgid"]),
            "started_by_benchmark": True,
        }
        for process in managed_processes
    }
    for process in local_processes:
        pid = int(process["pid"])
        if pid in managed_by_pid:
            managed = managed_by_pid[pid]
            managed["labels"] = sorted(set(managed["labels"]) | set(process["labels"]))
            managed["urls"] = sorted(set(managed["urls"]) | set(process["urls"]))
            managed["ports"] = sorted(set(managed["ports"]) | set(process["ports"]))
        else:
            managed_by_pid[pid] = process

    local_processes = sorted(managed_by_pid.values(), key=lambda item: int(item["pid"]))
    if not local_processes:
        return

    console.print("\n[bold yellow]Local benchmark targets still running[/bold yellow]")
    for process in local_processes:
        labels = ", ".join(process["labels"])
        ports = ", ".join(str(port) for port in process["ports"])
        command = process.get("command") or "<unknown command>"
        ownership = " [started by benchmark]" if process.get("started_by_benchmark") else ""
        console.print(
            f"- pid={process['pid']}{ownership} labels={labels} ports={ports}\n  command: {command}"
        )

    if not click.confirm("Kill detected local target processes now?", default=False):
        console.print("[yellow]Leaving local benchmark target processes running.[/yellow]")
        return

    managed_pgids = [int(process["pgid"]) for process in local_processes if process.get("pgid")]
    unmanaged_pids = [
        int(process["pid"])
        for process in local_processes
        if not process.get("started_by_benchmark")
    ]
    managed_result = (
        _terminate_process_groups(managed_pgids)
        if managed_pgids
        else {
            "terminated": [],
            "killed": [],
            "failed": [],
        }
    )
    unmanaged_result = (
        _terminate_processes(unmanaged_pids)
        if unmanaged_pids
        else {
            "terminated": [],
            "killed": [],
            "failed": [],
        }
    )
    console.print(
        "[green]Cleanup complete[/green]: "
        "terminated="
        f"{sorted(managed_result['terminated'] + unmanaged_result['terminated'])} "
        "killed="
        f"{sorted(managed_result['killed'] + unmanaged_result['killed'])} "
        "failed="
        f"{sorted(managed_result['failed'] + unmanaged_result['failed'])}"
    )


def _run_compare_command(
    *,
    targets: tuple[str, ...],
    model: str,
    hardware_family: str,
    batch_sizes: tuple[int, ...],
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
    max_seq_len: int | None,
    max_output_tokens: int | None,
    output_dir: str | None,
    prompt_cleanup: bool | None,
    target_commands: dict[str, str] | None = None,
    header: str = "sageLLM Endpoint Compare",
) -> Path:
    """Run a multi-endpoint compare and write artifacts."""
    if len(targets) < 2:
        raise click.BadParameter("Repeat --target at least twice to compare multiple endpoints.")

    parsed_targets = [_parse_compare_target(spec) for spec in targets]
    managed_processes = _maybe_start_local_targets(
        parsed_targets=parsed_targets,
        target_commands=target_commands or {},
        api_key=api_key,
        request_timeout=request_timeout,
        server_wait_s=server_wait_s,
    )
    compare_output_dir = _create_compare_output_dir(output_dir)
    compare_output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]{header}[/bold cyan]")
    console.print(f"Model: {model}")
    console.print(f"Targets: {len(parsed_targets)}")
    console.print(f"Output: {compare_output_dir}\n")

    target_results: list[dict[str, object]] = []

    for label, url in parsed_targets:
        console.print(f"[bold]Running target:[/bold] {label} -> {url}")
        target_result = _run_compare_target(
            label=label,
            url=url,
            model=model,
            hardware_family=hardware_family,
            batch_sizes=batch_sizes,
            api_key=api_key,
            request_timeout=request_timeout,
            server_wait_s=server_wait_s,
            max_seq_len=max_seq_len,
            max_output_tokens=max_output_tokens,
            output_dir=compare_output_dir,
        )
        target_results.append(target_result)
        summary = target_result["summary"]
        console.print(
            f"[green]✓[/green] {label}: TTFT={summary['avg_ttft_ms']:.2f}ms, "
            f"TBT={summary['avg_tbt_ms']:.2f}ms, TPS={summary['avg_throughput_tps']:.2f}"
        )

    _write_compare_summary_artifacts(
        compare_output_dir=compare_output_dir,
        model=model,
        batch_sizes=list(batch_sizes),
        target_results=target_results,
    )

    comparison_json = compare_output_dir / "comparison.json"
    comparison_md = compare_output_dir / "comparison.md"

    console.print("\n[bold green]✓ Compare completed[/bold green]")
    console.print(f"Comparison JSON: {comparison_json}")
    console.print(f"Comparison Markdown: {comparison_md}")
    for target_result in target_results:
        console.print(f"Parity Artifact ({target_result['label']}): {target_result['parity_json']}")
    _maybe_prompt_cleanup_local_targets(
        parsed_targets,
        prompt_cleanup=prompt_cleanup,
        managed_processes=managed_processes,
    )
    return compare_output_dir


def _resolve_local_benchmark_root() -> Path | None:
    """Return the local repo root when running from a source checkout."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return None


def _resolve_benchmark_extra_install_target(extra_name: str) -> str:
    """Resolve the install target for a benchmark extra."""
    local_root = _resolve_local_benchmark_root()
    if local_root is not None:
        return f"{local_root}[{extra_name}]"
    return f"isagellm-benchmark[{extra_name}]"


def _run_checked_command(command: list[str], input_text: str | None = None) -> None:
    """Run a command and raise a click-friendly error on failure."""
    console.print(f"[dim]$ {' '.join(shlex.quote(part) for part in command)}[/dim]")
    kwargs: dict[str, object] = {"check": True}
    if input_text is not None:
        kwargs["input"] = input_text
        kwargs["text"] = True

    try:
        subprocess.run(command, **kwargs)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc


def _get_vllm_compare_smoke_test_script() -> str:
    """Return the Ascend smoke test script used by install-ascend."""
    return """import torch, torch_npu

print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('npu_available', torch.npu.is_available())

if not torch.npu.is_available():
    raise RuntimeError('torch.npu.is_available() == False')

torch.npu.set_device('npu:0')
x = torch.ones(1, device='npu')
print('tensor_ok', (x + 1).cpu().tolist())
"""


def normalize_model_name(model_path: str) -> str:
    """Normalize model path to directory name.

    Args:
        model_path: Model path or HuggingFace repo ID

    Returns:
        Normalized model name for directory

    Examples:
        sshleifer/tiny-gpt2 → tiny-gpt2
        Qwen/Qwen2-7B-Instruct → Qwen2-7B-Instruct
        /path/to/model → model
    """
    # Remove leading/trailing slashes
    model_path = model_path.strip("/")

    # If it's a HuggingFace repo (contains /), take the last part
    if "/" in model_path:
        model_path = model_path.split("/")[-1]

    # If it's a local path, take basename
    if model_path.startswith("/") or model_path.startswith("./"):
        model_path = Path(model_path).name

    # Replace special characters
    model_path = model_path.replace(" ", "-").replace("_", "-")

    return model_path


def create_output_directory(
    backend: str,
    model: str,
    workload: str,
    custom_path: str | None = None,
) -> tuple[Path, dict]:
    """Create hierarchical output directory.

    Directory structure: outputs/<backend>/<model>/<workload_YYYYMMDD_NNN>/

    Args:
        backend: Backend name (cpu, cuda, vllm, etc.)
        model: Model name/path
        workload: Workload type (m1, short, long, stress)
        custom_path: User-specified output path (optional)

    Returns:
        Tuple of (output_path, metadata_dict)
    """
    if custom_path:
        # User specified path - use as-is
        output_dir = Path(custom_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, {"custom_output": True}

    # Standard hierarchical structure
    outputs_root = Path("outputs")
    model_name = normalize_model_name(model)

    # Create backend/model directory
    backend_model_dir = outputs_root / backend / model_name
    backend_model_dir.mkdir(parents=True, exist_ok=True)

    # Find next sequence number for today
    today = datetime.now().strftime("%Y%m%d")
    existing_runs = list(backend_model_dir.glob(f"{workload}_{today}_*"))
    seq_num = len(existing_runs) + 1

    # Create run directory: workload_YYYYMMDD_NNN
    run_id = f"{workload}_{today}_{seq_num:03d}"
    output_dir = backend_model_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create/update 'latest' symlink in backend/model directory
    latest_link = backend_model_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()

    try:
        # Create relative symlink
        latest_link.symlink_to(run_id)
    except OSError:
        # Windows may not support symlinks
        pass

    metadata = {
        "run_id": run_id,
        "backend": backend,
        "model": model_name,
        "workload": workload,
        "date": today,
        "sequence": seq_num,
    }

    return output_dir, metadata


def save_run_config(
    output_dir: Path,
    backend: str,
    model: str,
    workload: str,
    dataset: str,
    num_samples: int,
    metadata: dict,
) -> None:
    """Save run configuration to config.json.

    Args:
        output_dir: Output directory path
        backend: Backend name
        model: Model name/path
        workload: Workload type
        dataset: Dataset name
        num_samples: Number of samples
        metadata: Additional metadata from create_output_directory
    """
    versions = collect_installed_versions()

    config = {
        **metadata,
        "timestamp": datetime.now().isoformat(),
        "model_path": model,  # Original model path
        "dataset": dataset,
        "num_samples": num_samples,
        "versions": versions,
    }

    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    console.print(f"[dim]Saved config: {config_file}[/dim]")


def collect_installed_versions() -> dict[str, str]:
    """Collect installed sageLLM component versions from Python environment.

    Returns:
        Mapping of internal component keys to installed package versions.
    """
    try:
        import importlib.metadata
    except Exception:
        return {}

    package_map = {
        "sagellm": "isagellm",
        "sagellm_benchmark": "isagellm-benchmark",
        "sagellm_protocol": "isagellm-protocol",
        "sagellm_backend": "isagellm-backend",
        "sagellm_core": "isagellm-core",
        "sagellm_kv_cache": "isagellm-kv-cache",
        "sagellm_control_plane": "isagellm-control-plane",
        "sagellm_gateway": "isagellm-gateway",
        "sagellm_comm": "isagellm-comm",
        "sagellm_compression": "isagellm-compression",
    }

    versions: dict[str, str] = {}
    for key, package_name in package_map.items():
        try:
            versions[key] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue

    return versions


@click.group()
@click.version_option(version="0.1.0", prog_name="sagellm-benchmark")
def main() -> None:
    """sageLLM Benchmark Suite - M1 Demo Contract Validation."""
    pass


@main.group("parity-gate")
def parity_gate_group() -> None:
    """Inspect or evaluate parity gates."""


@parity_gate_group.command("print-default")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Optional path to write the default parity gate JSON.",
)
def parity_gate_print_default(output: str | None) -> None:
    """Print the default CUDA decode parity gate definition."""
    gate = build_default_cuda_decode_gate()
    payload = gate.model_dump_json(indent=2)
    if output is None:
        click.echo(payload)
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n")
    console.print(f"[green]✓[/green] Wrote default parity gate: {output_path}")


@parity_gate_group.command("evaluate")
@click.option(
    "--candidate",
    required=True,
    type=click.Path(exists=True),
    help="Candidate parity artifact or compare-record e2e payload.",
)
@click.option(
    "--reference",
    "references",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Reference parity artifact or compare-record e2e payload. Repeat for multiple engines.",
)
@click.option(
    "--gate-json",
    type=click.Path(exists=True),
    default=None,
    help="Optional custom parity gate JSON. Defaults to the built-in CUDA decode gate.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Optional path to write the evaluation JSON.",
)
def parity_gate_evaluate(
    candidate: str,
    references: tuple[str, ...],
    gate_json: str | None,
    output: str | None,
) -> None:
    """Evaluate a candidate artifact against the parity gate."""
    gate = (
        DecodeParityGate.model_validate_json(Path(gate_json).read_text())
        if gate_json is not None
        else build_default_cuda_decode_gate()
    )
    candidate_artifact = load_parity_run_artifact(candidate)
    reference_artifacts = [load_parity_run_artifact(path) for path in references]
    evaluation = evaluate_parity_gate(gate, candidate_artifact, reference_artifacts)

    console.print("[bold cyan]Parity Gate Evaluation[/bold cyan]")
    console.print(f"Gate: {evaluation.gate_id}")
    console.print(f"Candidate: {evaluation.candidate_label}")
    console.print(f"Passed: {'yes' if evaluation.passed else 'no'}")
    scenario_groups: dict[str, list[object]] = {}
    for result in evaluation.results:
        scenario_groups.setdefault(result.scenario_name, []).append(result)
    for scenario_name, scenario_results in scenario_groups.items():
        console.print(f"- {scenario_name}")
        for result in scenario_results:
            console.print(f"  * {result.category.value}: {result.message}")

    payload = evaluation.model_dump_json(indent=2)
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
        console.print(f"[green]✓[/green] Wrote parity evaluation: {output_path}")


@parity_gate_group.command("convert-core-telemetry")
@click.option(
    "--input-json",
    required=True,
    type=click.Path(exists=True),
    help="Path to a full LLMEngine.get_info() dump or performance_mainline.explicit_decode JSON.",
)
@click.option(
    "--label",
    required=True,
    help="Stable benchmark label for the captured endpoint, e.g. sagellm_before.",
)
@click.option(
    "--model",
    required=True,
    help="Model name associated with this telemetry capture.",
)
@click.option(
    "--hardware-family",
    required=True,
    help="Hardware family for this capture, e.g. cuda or ascend.",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Path to write the normalized benchmark telemetry artifact.",
)
def parity_gate_convert_core_telemetry(
    input_json: str,
    label: str,
    model: str,
    hardware_family: str,
    output: str,
) -> None:
    """Convert sagellm-core explicit decode telemetry into a stable benchmark artifact."""
    input_path = Path(input_json)
    artifact = build_core_decode_telemetry_artifact(
        json.loads(input_path.read_text()),
        label=label,
        model=model,
        hardware_family=hardware_family,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(artifact.model_dump_json(indent=2) + "\n")
    console.print(f"[green]✓[/green] Wrote core decode telemetry artifact: {output_path}")


@main.command("compare-record")
@click.option("--label", required=True, help="Label for this target capture, e.g. sagellm.")
@click.option("--url", required=True, help="OpenAI-compatible endpoint URL to benchmark.")
@click.option(
    "--hardware-family",
    required=True,
    help="Hardware family used by this live compare run, e.g. cuda or ascend.",
)
@click.option(
    "--model",
    type=str,
    default="Qwen/Qwen2.5-0.5B-Instruct",
    show_default=True,
    help="Requested model name for the benchmark run.",
)
@click.option(
    "--batch-size",
    "batch_sizes",
    multiple=True,
    type=int,
    default=(1, 2, 4),
    show_default=True,
    help="Batch sizes to benchmark. Repeat for multiple values.",
)
@click.option(
    "--api-key",
    type=str,
    default="sagellm-benchmark",
    show_default=True,
    help="API key used for the endpoint.",
)
@click.option(
    "--request-timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--server-wait",
    "server_wait_s",
    type=float,
    default=30.0,
    show_default=True,
    help="Max seconds to wait for the endpoint to become ready.",
)
@click.option(
    "--max-seq-len",
    type=int,
    default=None,
    help="Override the detected maximum sequence length.",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=64,
    show_default=True,
    help="Hard cap on output tokens for each request.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Directory to save per-target artifacts (default: benchmark_results/compare-record_<timestamp>).",
)
def compare_record(
    label: str,
    url: str,
    hardware_family: str,
    model: str,
    batch_sizes: tuple[int, ...],
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
    max_seq_len: int | None,
    max_output_tokens: int | None,
    output_dir: str | None,
) -> None:
    """Capture a single target's live benchmark result for later offline comparison."""
    compare_output_dir = _create_compare_output_dir(output_dir, prefix="compare-record")
    compare_output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]sageLLM Compare Record[/bold cyan]")
    console.print(f"Label: {label}")
    console.print(f"URL: {url}")
    console.print(f"Model: {model}")
    console.print(f"Output: {compare_output_dir}\n")

    target_result = _run_compare_target(
        label=label,
        url=url,
        model=model,
        hardware_family=hardware_family,
        batch_sizes=batch_sizes,
        api_key=api_key,
        request_timeout=request_timeout,
        server_wait_s=server_wait_s,
        max_seq_len=max_seq_len,
        max_output_tokens=max_output_tokens,
        output_dir=compare_output_dir,
    )
    summary = target_result["summary"]

    console.print(
        f"[green]✓[/green] {label}: TTFT={summary['avg_ttft_ms']:.2f}ms, "
        f"TBT={summary['avg_tbt_ms']:.2f}ms, TPS={summary['avg_throughput_tps']:.2f}"
    )
    console.print(f"JSON: {target_result['json']}")
    console.print(f"Markdown: {target_result['markdown']}")
    console.print(f"Parity Artifact: {target_result['parity_json']}")


@main.command("validate-serving-consistency")
@click.option("--label", required=True, help="Label for this target capture, e.g. sagellm.")
@click.option("--url", required=True, help="OpenAI-compatible endpoint URL to validate.")
@click.option(
    "--hardware-family",
    required=True,
    help="Hardware family used by this live validation run, e.g. cuda or ascend.",
)
@click.option(
    "--model",
    type=str,
    default="Qwen/Qwen2.5-0.5B-Instruct",
    show_default=True,
    help="Requested model name for the validation run.",
)
@click.option(
    "--reference-artifact",
    required=True,
    type=click.Path(exists=True),
    help="Backend benchmark artifact used as the expected runtime conclusion.",
)
@click.option(
    "--batch-size",
    "batch_sizes",
    multiple=True,
    type=int,
    default=(1,),
    show_default=True,
    help="Small-batch decode sizes to validate. Repeat for multiple values.",
)
@click.option(
    "--api-key",
    type=str,
    default="sagellm-benchmark",
    show_default=True,
    help="API key used for the endpoint.",
)
@click.option(
    "--request-timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--server-wait",
    "server_wait_s",
    type=float,
    default=30.0,
    show_default=True,
    help="Max seconds to wait for the endpoint to become ready.",
)
@click.option(
    "--max-seq-len",
    type=int,
    default=None,
    help="Override the detected maximum sequence length.",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=64,
    show_default=True,
    help="Hard cap on output tokens for each request.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help=(
        "Directory to save validation artifacts "
        "(default: benchmark_results/validate-serving-consistency_<timestamp>)."
    ),
)
def validate_serving_consistency(
    label: str,
    url: str,
    hardware_family: str,
    model: str,
    reference_artifact: str,
    batch_sizes: tuple[int, ...],
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
    max_seq_len: int | None,
    max_output_tokens: int,
    output_dir: str | None,
) -> None:
    """Run a minimal live decode retest and fail fast on runtime evidence mismatches."""
    validation_output_dir = _create_compare_output_dir(
        output_dir,
        prefix="validate-serving-consistency",
    )
    validation_output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]sageLLM Serving Consistency Validation[/bold cyan]")
    console.print(f"Label: {label}")
    console.print(f"URL: {url}")
    console.print(f"Model: {model}")
    console.print(f"Reference Artifact: {reference_artifact}")
    console.print(f"Output: {validation_output_dir}\n")

    target_result = _run_compare_target(
        label=label,
        url=url,
        model=model,
        hardware_family=hardware_family,
        batch_sizes=batch_sizes,
        api_key=api_key,
        request_timeout=request_timeout,
        server_wait_s=server_wait_s,
        max_seq_len=max_seq_len,
        max_output_tokens=max_output_tokens,
        output_dir=validation_output_dir,
    )
    try:
        report = build_live_runtime_consistency_report(
            label=label,
            url=url,
            model=model,
            hardware_family=hardware_family,
            requested_batch_sizes=list(batch_sizes),
            target_payload=target_result["payload"],
            runtime_artifacts=target_result["runtime_artifacts"],
            reference_artifact_path=reference_artifact,
        )
    except ValueError as exc:
        report = {
            "schema_version": "live-runtime-consistency/v1",
            "passed": False,
            "label": label,
            "url": url,
            "model": model,
            "hardware_family": hardware_family,
            "validation_batch_sizes": list(batch_sizes),
            "successful_live_batch_sizes": sorted(
                {
                    int(row.get("batch_size", 0))
                    for row in target_result["payload"].get("rows", [])
                    if isinstance(row, dict) and int(row.get("successful_requests", 0) or 0) > 0
                }
            ),
            "observed_batch_size": None,
            "reference_artifact": str(reference_artifact),
            "runtime_artifacts": dict(target_result["runtime_artifacts"]),
            "observed": {},
            "reference": {},
            "findings": [
                {
                    "code": "precondition-failure",
                    "message": str(exc),
                }
            ],
        }

    report_path = validation_output_dir / f"{_slugify_filename(label)}_runtime_consistency.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    observed = report["observed"]
    console.print(
        "Observed: "
        f"attention={observed.get('attention_selected_implementation')}, "
        f"adjacent={observed.get('adjacent_selected_implementation')}, "
        f"batch={report['observed_batch_size']}"
    )
    console.print(f"Validation Report: {report_path}")

    if not report["passed"]:
        finding_lines = "\n".join(
            f"- {finding['code']}: {finding['message']}" for finding in report["findings"]
        )
        raise click.ClickException("Live serving consistency validation failed:\n" + finding_lines)

    console.print(
        "[bold green]✓ Runtime evidence is consistent across /info, telemetry, and artifact[/bold green]"
    )


@main.command("compare-offline")
@click.option(
    "--result",
    "results",
    multiple=True,
    required=True,
    help=(
        "Captured result in LABEL=PATH format. Repeat to compare multiple offline captures, "
        "e.g. --result sagellm=./captures/sagellm.json --result vllm=./captures/vllm.json"
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Directory to save offline comparison artifacts (default: benchmark_results/compare_<timestamp>).",
)
def compare_offline(results: tuple[str, ...], output_dir: str | None) -> None:
    """Build comparison artifacts from previously captured single-target results."""
    if len(results) < 2:
        raise click.BadParameter("Repeat --result at least twice to compare multiple captures.")

    parsed_results = [_parse_label_path(spec) for spec in results]
    target_results: list[dict[str, object]] = []
    models_seen: list[str] = []
    batch_sizes_seen: list[list[int]] = []

    for label, path in parsed_results:
        payload = _load_compare_result_payload(label, path)
        models = payload.get("models") or []
        batch_sizes = payload.get("batch_sizes") or []
        model_name = str(models[0]) if models else "unknown"
        models_seen.append(model_name)
        batch_sizes_seen.append([int(size) for size in batch_sizes])
        target_results.append(
            {
                "label": payload["label"],
                "url": payload["url"],
                "summary": payload["summary"],
                "json": str(Path(path)),
                "markdown": str(Path(path).with_suffix(".md")),
            }
        )

    if len(set(models_seen)) != 1:
        raise click.ClickException(
            f"Offline compare requires the same model across captures, got: {sorted(set(models_seen))}"
        )
    normalized_batch_sizes = {tuple(sizes) for sizes in batch_sizes_seen}
    if len(normalized_batch_sizes) != 1:
        raise click.ClickException(
            f"Offline compare requires the same batch sizes across captures, got: {sorted(normalized_batch_sizes)}"
        )

    compare_output_dir = _create_compare_output_dir(output_dir)
    compare_output_dir.mkdir(parents=True, exist_ok=True)
    compare_result = _write_compare_summary_artifacts(
        compare_output_dir=compare_output_dir,
        model=models_seen[0],
        batch_sizes=list(next(iter(normalized_batch_sizes))),
        target_results=target_results,
    )

    comparison_json = compare_output_dir / "comparison.json"
    comparison_md = compare_output_dir / "comparison.md"
    console.print("[bold cyan]sageLLM Offline Compare[/bold cyan]")
    console.print(f"Baseline: {compare_result['baseline']}")
    console.print(f"Targets: {len(target_results)}")
    console.print(f"Comparison JSON: {comparison_json}")
    console.print(f"Comparison Markdown: {comparison_md}")


@main.command()
@click.option(
    "--workload",
    type=click.Choice(
        [
            "all",
            "query",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5",
            "Q6",
            "Q7",
            "Q8",
            "streaming",
            "batch",
            "mixed",
        ],
        case_sensitive=False,
    ),
    default="all",
    help="Workload type to run (Q1-Q8 query workloads, or 'all' for full suite).",
)
@click.option(
    "--backend",
    type=click.Choice(["cpu", "lmdeploy", "vllm"]),
    default="cpu",
    help="Backend engine to use.",
)
@click.option(
    "--model",
    type=str,
    default="sshleifer/tiny-gpt2",
    help="Model path (for CPU backend).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory (default: outputs/<backend>/<model>/<workload_date_seq>/).",
)
@click.option(
    "--mode",
    type=click.Choice(["batch", "traffic"]),
    default="traffic",
    help="Benchmark mode: 'batch' for offline throughput (all requests at once), 'traffic' for arrival pattern simulation.",
)
@click.option(
    "--output-json",
    type=click.Path(),
    default=None,
    help="Path to save JSON output (in addition to default location).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging.",
)
@click.option(
    "--dataset",
    type=click.Choice(["default", "sharegpt", "synthetic"]),
    default="default",
    help="Dataset to use for prompts (default: hardcoded prompts, sharegpt: HuggingFace ShareGPT).",
)
@click.option(
    "--num-samples",
    type=int,
    default=5,
    help="Number of samples to use from dataset (ignored for 'default').",
)
def run(
    workload: str,
    backend: str,
    model: str | None,
    output: str,
    mode: str,
    output_json: str | None,
    verbose: bool,
    dataset: str,
    num_samples: int,
) -> None:
    """Run benchmark workloads."""
    console.print("[bold cyan]sageLLM Benchmark[/bold cyan]")
    console.print(f"Workload: {workload}")
    console.print(f"Backend: {backend}")
    console.print(f"Model: {model}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Mode: {mode}")

    # Create hierarchical output directory
    output_dir, metadata = create_output_directory(backend, model or "default", workload, output)
    console.print(f"[bold green]Output:[/bold green] {output_dir}\n")

    # Import LLMEngine
    try:
        from sagellm_core import LLMEngine, LLMEngineConfig
    except ImportError:
        console.print("[bold red]Error:[/bold red] isagellm-core not installed.")
        console.print("Install with: pip install isagellm-core")
        sys.exit(1)

    # Determine workloads to run
    from sagellm_benchmark.workloads import get_workloads_by_selector

    # Load dataset if needed
    dataset_instance = None
    if dataset == "sharegpt":
        console.print("Loading ShareGPT dataset from HuggingFace...")
        from sagellm_benchmark.datasets import ShareGPTDataset

        try:
            dataset_instance = ShareGPTDataset.from_huggingface(
                repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
                split="train[:1000]",  # Load first 1000 for speed
                min_prompt_len=50,
                max_prompt_len=5000,
                seed=42,
            )
            console.print(f"✓ Loaded {len(dataset_instance)} prompts from ShareGPT")
        except Exception as e:
            console.print(f"[bold red]Error loading ShareGPT:[/bold red] {e}")
            console.print("Falling back to default prompts")
            dataset_instance = None
    elif dataset == "synthetic":
        console.print("Using synthetic ShareGPT-style prompts...")
        from sagellm_benchmark.datasets import SyntheticShareGPTDataset

        dataset_instance = SyntheticShareGPTDataset(seed=42)
        console.print("✓ Synthetic dataset ready")

    try:
        workloads = get_workloads_by_selector(workload)
    except ValueError:
        console.print(f"[bold red]Unknown workload:[/bold red] {workload}")
        sys.exit(1)

    # Override num_requests if using dataset
    if dataset_instance is not None:
        for w in workloads:
            w.num_requests = num_samples

    # Create engine using LLMEngine
    if backend == "cpu":
        try:
            from sagellm_core import LLMEngine, LLMEngineConfig
        except ImportError:
            console.print("[bold red]Error:[/bold red] isagellm-core not installed.")
            console.print("Install with: pip install isagellm-core")
            sys.exit(1)

        # Create LLMEngine config
        engine_config = LLMEngineConfig(
            model_path=model,
            backend_type="cpu",  # Use CPU backend
            comm_type="gloo",  # Not used in single-device mode
            max_batch_size=32,
            max_model_len=4096,
            max_new_tokens=128,
            trust_remote_code=True,
        )

        # Create engine
        engine = LLMEngine(engine_config)

        # Start engine
        console.print(f"[dim]Starting engine with model: {model}[/dim]")
        asyncio.run(engine.start())
        console.print("[green]✓[/green] Engine started\n")

    elif backend in ["lmdeploy", "vllm"]:
        console.print(f"[bold red]Backend not yet implemented:[/bold red] {backend}")
        console.print("Available: cpu")
        console.print("[dim]lmdeploy and vllm support coming soon[/dim]")
        sys.exit(1)
    else:
        console.print(f"[bold red]Unknown backend:[/bold red] {backend}")
        console.print("Available: cpu")
        sys.exit(1)

    # Run benchmark
    from sagellm_benchmark.runner import BenchmarkConfig, BenchmarkRunner

    bench_config = BenchmarkConfig(
        engine=engine,
        workloads=workloads,
        output_dir=output_dir,
        verbose=verbose,
        dataset=dataset_instance,  # Pass dataset to runner
        mode=mode,  # Pass benchmark mode
    )

    # Save run configuration
    save_run_config(
        output_dir, backend, model or "default", workload, dataset, num_samples, metadata
    )

    runner = BenchmarkRunner(bench_config)

    console.print("\n[bold green]Starting benchmark...[/bold green]")
    console.print(f"Workloads: {len(workloads)}\n")

    try:
        results = asyncio.run(runner.run())

        # Display summary
        console.print("\n[bold green]✓ Benchmark completed![/bold green]\n")
        _display_results(results)

        console.print(f"\n[bold]Results saved to:[/bold] {output_dir}")

        # Save to custom JSON output if specified
        if output_json:
            import json

            output_json_path = Path(output_json)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert results to serializable format
            json_results = {}
            for name, metrics in results.items():
                from dataclasses import asdict

                json_results[name] = asdict(metrics)

            with open(output_json_path, "w") as f:
                json.dump(json_results, f, indent=2)

            console.print(f"[bold]Additional JSON output:[/bold] {output_json_path}")

        # Show latest link if not custom output
        if not metadata.get("custom_output"):
            latest_path = output_dir.parent / "latest"
            console.print(f"[dim]Latest results: {latest_path}[/dim]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Benchmark failed:[/bold red] {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option(
    "--type",
    "benchmark_type",
    type=click.Choice(["operator", "e2e"]),
    default="operator",
    help="Performance benchmark type.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="Execution device for operator benchmark (e.g., cpu, cuda).",
)
@click.option(
    "--iterations",
    type=int,
    default=20,
    help="Benchmark iterations for operator benchmark.",
)
@click.option(
    "--warmup",
    type=int,
    default=5,
    help="Warmup iterations for operator benchmark.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    default=("Qwen/Qwen2-7B-Instruct",),
    help="Model(s) for e2e benchmarks. Repeat for multiple models.",
)
@click.option(
    "--batch-size",
    "batch_sizes",
    multiple=True,
    type=int,
    default=(1, 4, 8),
    help="Batch sizes for e2e benchmark. Repeat for multiple values.",
)
@click.option(
    "--precision",
    "precisions",
    multiple=True,
    default=("fp16", "int8"),
    help="Precisions for e2e benchmark. Repeat for multiple values.",
)
@click.option(
    "--simulate/--live",
    default=True,
    help="Run e2e benchmark in deterministic simulation mode (default) or live mode.",
)
@click.option(
    "--backend-url",
    type=str,
    default="http://localhost:8000/v1",
    show_default=True,
    help="API base URL for live e2e benchmark mode (OpenAI-compatible endpoint).",
)
@click.option(
    "--api-key",
    type=str,
    default="sagellm-benchmark",
    show_default=True,
    help="API key for live e2e benchmark mode.",
)
@click.option(
    "--request-timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="Per-request timeout in seconds for live e2e mode.",
)
@click.option(
    "--server-wait",
    "server_wait_s",
    type=float,
    default=30.0,
    show_default=True,
    help="Max seconds to wait for the API server to become ready in live mode.",
)
@click.option(
    "--max-seq-len",
    "max_seq_len",
    type=int,
    default=None,
    help=(
        "Maximum sequence length (prompt + output tokens) the model supports. "
        "Auto-detected if not set. Used in live mode to clamp prompts so they "
        "never exceed the model's context window."
    ),
)
@click.option(
    "--max-output-tokens",
    "max_output_tokens",
    type=int,
    default=None,
    help=(
        "Hard cap on output tokens per request in live e2e mode. "
        "Use this for CPU/slow models where the full scenario output length would "
        "exceed the request timeout. E.g. '--max-output-tokens 16' for tiny CPU models."
    ),
)
@click.option(
    "--output-json",
    type=click.Path(),
    default="./benchmark_results/perf_results.json",
    help="Path to save performance JSON result.",
)
@click.option(
    "--output-markdown",
    type=click.Path(),
    default="./benchmark_results/perf_report.md",
    help="Path to save performance markdown report.",
)
@click.option(
    "--plot/--no-plot",
    default=False,
    help="Generate performance charts.",
)
@click.option(
    "--plot-format",
    "plot_formats",
    multiple=True,
    type=click.Choice(["png", "pdf"]),
    default=("png",),
    help="Plot output format(s). Repeat for multiple formats.",
)
@click.option(
    "--theme",
    type=click.Choice(["light", "dark"]),
    default="light",
    help="Chart theme.",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="Chart output DPI.",
)
def perf(
    benchmark_type: str,
    device: str,
    iterations: int,
    warmup: int,
    models: tuple[str, ...],
    batch_sizes: tuple[int, ...],
    precisions: tuple[str, ...],
    simulate: bool,
    backend_url: str,
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
    max_seq_len: int | None,
    max_output_tokens: int | None,
    output_json: str,
    output_markdown: str,
    plot: bool,
    plot_formats: tuple[str, ...],
    theme: str,
    dpi: int,
) -> None:
    """Run performance benchmarks (operator/e2e) migrated from sagellm-core."""
    console.print("[bold cyan]sageLLM Performance Benchmark[/bold cyan]")
    console.print(f"Type: {benchmark_type}")

    if benchmark_type == "operator":
        from sagellm_benchmark.performance.benchmark_utils import format_comparison_table
        from sagellm_benchmark.performance.operator_benchmarks import run_operator_benchmarks

        comparisons = run_operator_benchmarks(device=device, iterations=iterations, warmup=warmup)
        markdown = "# Operator Benchmark Report\n\n" + format_comparison_table(comparisons)
        result_data = {
            "kind": "operator",
            "device": device,
            "iterations": iterations,
            "warmup": warmup,
            "comparisons": comparisons,
        }
        console.print("\n" + format_comparison_table(comparisons))
    else:
        from sagellm_benchmark.performance.model_benchmarks import (
            run_e2e_model_benchmarks,
            summarize_e2e_rows,
        )

        if simulate:
            console.print("[dim]Mode: simulate (deterministic)[/dim]")
        else:
            import logging as _logging

            # Enable INFO logging for live mode so server wait/discovery messages are visible
            _logging.basicConfig(
                level=_logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
            console.print(
                f"[bold yellow]Mode: live — sending real requests to {backend_url}[/bold yellow]"
            )
            console.print(
                f"[dim]Models: {', '.join(models)} | "
                f"batch sizes: {', '.join(str(b) for b in batch_sizes)} | "
                f"timeout: {request_timeout:.0f}s/req[/dim]"
            )

        rows = run_e2e_model_benchmarks(
            models=list(models),
            batch_sizes=list(batch_sizes),
            precisions=list(precisions),
            simulate=simulate,
            backend_url=backend_url,
            api_key=api_key,
            request_timeout=request_timeout,
            server_wait_s=server_wait_s,
            max_seq_len=max_seq_len,
            max_output_tokens=max_output_tokens,
        )
        summary = summarize_e2e_rows(rows)
        result_data = {
            "kind": "e2e",
            "simulate": simulate,
            "models": list(models),
            "batch_sizes": list(batch_sizes),
            "precisions": list(precisions),
            "summary": summary,
            "rows": rows,
        }
        markdown = _format_e2e_markdown(result_data)
        _display_perf_e2e_table(result_data)

    if plot:
        from sagellm_benchmark.performance.plotting import generate_perf_charts

        plot_output_dir = Path(output_markdown).parent / "plots"
        plot_paths = generate_perf_charts(
            result_data,
            output_dir=plot_output_dir,
            formats=list(plot_formats),
            theme=theme,
            dpi=dpi,
        )
        result_data["plots"] = plot_paths
        console.print("\n[bold]Generated plots:[/bold]")
        for path in plot_paths:
            console.print(f"- {path}")

    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(result_data, f, indent=2)

    output_md_path = Path(output_markdown)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w") as f:
        f.write(markdown + "\n")

    console.print("\n[bold green]✓ Performance benchmark completed[/bold green]")
    console.print(f"JSON: {output_json_path}")
    console.print(f"Markdown: {output_md_path}")


@main.command()
@click.option(
    "--target",
    "targets",
    multiple=True,
    required=True,
    help=(
        "Comparison target in LABEL=URL format. Repeat to compare multiple OpenAI-compatible "
        "endpoints, e.g. --target sagellm=http://127.0.0.1:8000/v1 --target vllm=http://127.0.0.1:8000/v1"
    ),
)
@click.option(
    "--target-command",
    "target_commands",
    multiple=True,
    help="Optional local start command in LABEL=COMMAND format. If the target endpoint is not ready, benchmark will start it before compare.",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="Requested model name for the benchmark run.",
)
@click.option(
    "--hardware-family",
    required=True,
    help="Hardware family shared by the compared endpoints, e.g. cuda or ascend.",
)
@click.option(
    "--batch-size",
    "batch_sizes",
    multiple=True,
    type=int,
    default=(1, 2, 4),
    show_default=True,
    help="Batch sizes to benchmark. Repeat for multiple values.",
)
@click.option(
    "--api-key",
    type=str,
    default="sagellm-benchmark",
    show_default=True,
    help="API key used for all targets.",
)
@click.option(
    "--request-timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--server-wait",
    "server_wait_s",
    type=float,
    default=30.0,
    show_default=True,
    help="Max seconds to wait for each server to become ready.",
)
@click.option(
    "--max-seq-len",
    "max_seq_len",
    type=int,
    default=None,
    help="Override the detected maximum sequence length for all targets.",
)
@click.option(
    "--max-output-tokens",
    "max_output_tokens",
    type=int,
    default=None,
    help="Hard cap on output tokens for each request.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Directory to save compare artifacts (default: benchmark_results/compare_<timestamp>).",
)
@click.option(
    "--prompt-cleanup/--no-prompt-cleanup",
    default=None,
    help="Prompt to terminate local target processes after compare completes. Defaults to on in interactive terminals only.",
)
def compare(
    targets: tuple[str, ...],
    target_commands: tuple[str, ...],
    model: str,
    hardware_family: str,
    batch_sizes: tuple[int, ...],
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
    max_seq_len: int | None,
    max_output_tokens: int | None,
    output_dir: str | None,
    prompt_cleanup: bool | None,
) -> None:
    """Compare multiple inference endpoints through benchmark clients only."""
    _run_compare_command(
        targets=targets,
        target_commands=dict(_parse_label_command(spec) for spec in target_commands),
        model=model,
        hardware_family=hardware_family,
        batch_sizes=batch_sizes,
        api_key=api_key,
        request_timeout=request_timeout,
        server_wait_s=server_wait_s,
        max_seq_len=max_seq_len,
        max_output_tokens=max_output_tokens,
        output_dir=output_dir,
        prompt_cleanup=prompt_cleanup,
    )


@main.group("vllm-compare")
def vllm_compare() -> None:
    """Convenience commands for vLLM compare environment setup and runs."""
    pass


@vllm_compare.command("install-ascend")
@click.option(
    "--python-bin",
    envvar="BENCH_VLLM_ASCEND_PY",
    default="/opt/miniconda3/envs/bench-vllm-ascend/bin/python",
    show_default=True,
    help="Python executable for the vLLM Ascend compare environment.",
)
@click.option(
    "--sagellm-root",
    envvar="SAGELLM_ROOT",
    default="/home/user8/sagellm",
    show_default=True,
    help="Path to the sagellm repo that provides scripts/sagellm_with_ascend_env.sh.",
)
def vllm_compare_install_ascend(python_bin: str, sagellm_root: str) -> None:
    """Install the validated vLLM Ascend compare environment and run smoke checks."""
    python_path = Path(python_bin).expanduser()
    wrapper_path = Path(sagellm_root).expanduser() / "scripts" / "sagellm_with_ascend_env.sh"

    console.print("[bold cyan]vLLM Ascend Compare Setup[/bold cyan]")
    console.print(f"Python: {python_path}")
    console.print(f"sagellm root: {Path(sagellm_root).expanduser()}")

    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise click.ClickException(f"Python executable not found or not executable: {python_path}")

    if not wrapper_path.is_file() or not os.access(wrapper_path, os.X_OK):
        raise click.ClickException(f"Ascend wrapper not found or not executable: {wrapper_path}")

    benchmark_target = _resolve_benchmark_extra_install_target("vllm-ascend-client")
    pinned_packages = [
        "torch==2.7.1",
        "torch-npu==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        "transformers==4.57.1",
        "vllm-ascend==0.11.0",
    ]

    _run_checked_command([str(python_path), "-m", "pip", "install", "-U", benchmark_target])
    _run_checked_command([str(python_path), "-m", "pip", "install", "-U", *pinned_packages])
    _run_checked_command([str(python_path), "-m", "pip", "check"])
    _run_checked_command(
        [str(wrapper_path), str(python_path), "-"],
        input_text=_get_vllm_compare_smoke_test_script(),
    )

    console.print("\n[bold green]✓ vLLM Ascend compare environment is ready[/bold green]")


@vllm_compare.command("run")
@click.option(
    "--vllm-url",
    envvar="VLLM_COMPARE_VLLM_URL",
    default="http://127.0.0.1:8000/v1",
    show_default=True,
    help="vLLM OpenAI-compatible endpoint URL.",
)
@click.option(
    "--start-vllm-cmd",
    type=str,
    default=None,
    help="Optional local command to start the vLLM endpoint when it is not already running.",
)
@click.option(
    "--sagellm-url",
    envvar="VLLM_COMPARE_SAGELLM_URL",
    default="http://127.0.0.1:8901/v1",
    show_default=True,
    help="sageLLM OpenAI-compatible endpoint URL.",
)
@click.option(
    "--start-sagellm-cmd",
    type=str,
    default=None,
    help="Optional local command to start the sageLLM endpoint when it is not already running.",
)
@click.option(
    "--model",
    type=str,
    default="Qwen/Qwen2.5-0.5B-Instruct",
    show_default=True,
    help="Requested model name for the benchmark run.",
)
@click.option(
    "--hardware-family",
    required=True,
    help="Hardware family shared by sageLLM and vLLM for this compare run, e.g. cuda or ascend.",
)
@click.option(
    "--batch-size",
    "batch_sizes",
    multiple=True,
    type=int,
    default=(1, 2, 4),
    show_default=True,
    help="Batch sizes to benchmark. Repeat for multiple values.",
)
@click.option(
    "--api-key",
    type=str,
    default="sagellm-benchmark",
    show_default=True,
    help="API key used for both endpoints.",
)
@click.option(
    "--request-timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--server-wait",
    "server_wait_s",
    type=float,
    default=30.0,
    show_default=True,
    help="Max seconds to wait for each endpoint to become ready.",
)
@click.option(
    "--max-seq-len",
    type=int,
    default=None,
    help="Override the detected maximum sequence length for both endpoints.",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=64,
    show_default=True,
    help="Hard cap on output tokens for each request.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Directory to save compare artifacts (default: benchmark_results/compare_<timestamp>).",
)
@click.option(
    "--prompt-cleanup/--no-prompt-cleanup",
    default=None,
    help="Prompt to terminate local target processes after compare completes. Defaults to on in interactive terminals only.",
)
def vllm_compare_run(
    vllm_url: str,
    start_vllm_cmd: str | None,
    sagellm_url: str,
    start_sagellm_cmd: str | None,
    model: str,
    hardware_family: str,
    batch_sizes: tuple[int, ...],
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
    max_seq_len: int | None,
    max_output_tokens: int,
    output_dir: str | None,
    prompt_cleanup: bool | None,
) -> None:
    """Run the standard sageLLM vs vLLM endpoint comparison."""
    _run_compare_command(
        targets=(f"sagellm={sagellm_url}", f"vllm={vllm_url}"),
        target_commands={
            key: value
            for key, value in {
                "sagellm": start_sagellm_cmd,
                "vllm": start_vllm_cmd,
            }.items()
            if value
        },
        model=model,
        hardware_family=hardware_family,
        batch_sizes=batch_sizes,
        api_key=api_key,
        request_timeout=request_timeout,
        server_wait_s=server_wait_s,
        max_seq_len=max_seq_len,
        max_output_tokens=max_output_tokens,
        output_dir=output_dir,
        prompt_cleanup=prompt_cleanup,
        header="sageLLM vs vLLM Compare",
    )


@main.command()
@click.option(
    "--target",
    "targets",
    multiple=True,
    required=True,
    help=(
        "Comparison target in LABEL=URL format. Repeat to compare multiple OpenAI-compatible "
        "endpoints, e.g. --target sagellm=http://127.0.0.1:8901/v1 --target vllm=http://127.0.0.1:8000/v1"
    ),
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="Requested model name for the compare run.",
)
@click.option(
    "--prompt",
    type=str,
    default="请用一句话介绍你自己。",
    show_default=True,
    help="Prompt sent to each endpoint.",
)
@click.option(
    "--batch-size",
    "batch_sizes",
    multiple=True,
    type=int,
    default=(1, 2),
    show_default=True,
    help="Batch sizes to benchmark. Repeat for multiple values.",
)
@click.option(
    "--warmup-rounds",
    type=int,
    default=1,
    show_default=True,
    help="Warmup requests per target before measured runs.",
)
@click.option(
    "--rounds",
    type=int,
    default=1,
    show_default=True,
    help="Measured rounds per batch size.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=8,
    show_default=True,
    help="Max completion tokens for each request.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    show_default=True,
    help="Sampling temperature for each request.",
)
@click.option(
    "--api-key",
    type=str,
    default="sagellm-benchmark",
    show_default=True,
    help="API key used for all targets.",
)
@click.option(
    "--request-timeout",
    type=float,
    default=600.0,
    show_default=True,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help=(
        "Directory to save compare artifacts (default: "
        "benchmark_results/nonstream_compare_<timestamp>)."
    ),
)
def nonstream_compare(
    targets: tuple[str, ...],
    model: str,
    prompt: str,
    batch_sizes: tuple[int, ...],
    warmup_rounds: int,
    rounds: int,
    max_tokens: int,
    temperature: float,
    api_key: str,
    request_timeout: float,
    output_dir: str | None,
) -> None:
    """Compare non-stream chat completions across multiple endpoints."""
    try:
        config = NonStreamCompareConfig(
            targets=tuple(parse_target_spec(spec) for spec in targets),
            model=model,
            prompt=prompt,
            batch_sizes=batch_sizes,
            warmup_rounds=warmup_rounds,
            rounds=rounds,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            request_timeout=request_timeout,
            output_dir=output_dir,
        )
        compare_output_dir = run_nonstream_compare(config)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print("[bold green]Non-stream compare complete[/bold green]")
    console.print(f"Artifacts: {compare_output_dir}")


@main.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    default="./benchmark_results/benchmark_summary.json",
    help="Input summary JSON file.",
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "markdown"]),
    default="table",
    help="Output format.",
)
@click.option(
    "--plot/--no-plot",
    default=False,
    help="Generate charts when input is a perf JSON.",
)
@click.option(
    "--plot-format",
    "plot_formats",
    multiple=True,
    type=click.Choice(["png", "pdf"]),
    default=("png",),
    help="Plot output format(s). Repeat for multiple formats.",
)
@click.option(
    "--theme",
    type=click.Choice(["light", "dark"]),
    default="light",
    help="Chart theme.",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="Chart output DPI.",
)
def report(
    input: str,
    format: str,
    plot: bool,
    plot_formats: tuple[str, ...],
    theme: str,
    dpi: int,
) -> None:
    """Generate report from benchmark results."""
    try:
        with open(input) as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] File not found: {input}")
        console.print("Run benchmark first with: sagellm-benchmark run")
        sys.exit(1)
    except json.JSONDecodeError:
        console.print(f"[bold red]Error:[/bold red] Invalid JSON file: {input}")
        sys.exit(1)

    if data.get("kind") == "operator":
        if plot:
            _generate_plots_for_report(input, data, plot_formats, theme, dpi)
        _display_perf_operator_report(data, format)
        return
    if data.get("kind") == "e2e":
        if plot:
            _generate_plots_for_report(input, data, plot_formats, theme, dpi)
        _display_perf_e2e_report(data, format)
        return

    if format == "table":
        _display_summary_table(data)
    elif format == "json":
        console.print(json.dumps(data, indent=2))
    elif format == "markdown":
        _display_markdown(data)


def _display_perf_operator_report(data: dict, format: str) -> None:
    from sagellm_benchmark.performance.benchmark_utils import format_comparison_table

    if format == "json":
        console.print(json.dumps(data, indent=2))
        return

    markdown = "# Operator Benchmark Report\n\n" + format_comparison_table(data["comparisons"])
    if format == "markdown":
        console.print(markdown)
        return

    console.print("[bold cyan]Operator Benchmark Summary[/bold cyan]")
    console.print(f"Device: {data.get('device', 'unknown')}")
    console.print(format_comparison_table(data["comparisons"]))


def _display_perf_e2e_report(data: dict, format: str) -> None:
    if format == "json":
        console.print(json.dumps(data, indent=2))
        return
    if format == "markdown":
        console.print(_format_e2e_markdown(data))
        return
    _display_perf_e2e_table(data)


def _display_perf_e2e_table(data: dict) -> None:
    summary = data.get("summary", {})
    console.print("[bold cyan]E2E Benchmark Summary[/bold cyan]")
    console.print(f"Rows: {summary.get('total_rows', 0)}")
    console.print(f"Avg TTFT (ms): {summary.get('avg_ttft_ms', 0.0):.2f}")
    console.print(f"Avg TBT (ms): {summary.get('avg_tbt_ms', 0.0):.2f}")
    console.print(f"Output Throughput (tok/s): {summary.get('output_throughput_tps', 0.0):.2f}")
    console.print(
        f"Avg Per-Request Throughput (tok/s): {summary.get('avg_throughput_tps', 0.0):.2f}\n"
    )

    table = Table(title="E2E Scenario Results")
    table.add_column("Model", style="cyan")
    table.add_column("Scenario")
    table.add_column("Precision")
    table.add_column("Batch", justify="right")
    table.add_column("TTFT(ms)", justify="right")
    table.add_column("TBT(ms)", justify="right")
    table.add_column("TPS", justify="right")
    table.add_column("P95(ms)", justify="right")

    for row in data.get("rows", []):
        table.add_row(
            str(row.get("model", "")),
            str(row.get("scenario", "")),
            str(row.get("precision", "default")),
            str(row.get("batch_size", "")),
            f"{float(row.get('ttft_ms', 0.0)):.2f}",
            f"{float(row.get('tbt_ms', 0.0)):.2f}",
            f"{float(row.get('throughput_tps', 0.0)):.2f}",
            f"{float(row.get('latency_p95_ms', 0.0)):.2f}",
        )
    console.print(table)


def _format_e2e_markdown(data: dict) -> str:
    summary = data.get("summary", {})
    lines = [
        "# E2E Benchmark Report",
        "",
        "## Summary",
        f"- Rows: {summary.get('total_rows', 0)}",
        f"- Avg TTFT (ms): {summary.get('avg_ttft_ms', 0.0):.2f}",
        f"- Avg TBT (ms): {summary.get('avg_tbt_ms', 0.0):.2f}",
        f"- Output Throughput (tok/s): {summary.get('output_throughput_tps', 0.0):.2f}",
        f"- Avg Per-Request Throughput (tok/s): {summary.get('avg_throughput_tps', 0.0):.2f}",
        "",
        "## Results",
        "",
        "| Model | Scenario | Precision | Batch | TTFT(ms) | TBT(ms) | TPS | P95(ms) |",
        "|-------|----------|-----------|-------|----------|---------|-----|---------|",
    ]
    for row in data.get("rows", []):
        lines.append(
            f"| {row.get('model', '')} | {row.get('scenario', '')} | {row.get('precision', 'default')} | "
            f"{row.get('batch_size', '')} | "
            f"{float(row.get('ttft_ms', 0.0)):.2f} | {float(row.get('tbt_ms', 0.0)):.2f} | "
            f"{float(row.get('throughput_tps', 0.0)):.2f} | {float(row.get('latency_p95_ms', 0.0)):.2f} |"
        )
    return "\n".join(lines)


def _generate_plots_for_report(
    input_path: str,
    data: dict,
    plot_formats: tuple[str, ...],
    theme: str,
    dpi: int,
) -> None:
    from sagellm_benchmark.performance.plotting import generate_perf_charts

    output_dir = Path(input_path).parent / "plots"
    paths = generate_perf_charts(
        data,
        output_dir=output_dir,
        formats=list(plot_formats),
        theme=theme,
        dpi=dpi,
    )
    console.print("\n[bold]Generated plots:[/bold]")
    for path in paths:
        console.print(f"- {path}")


def _display_results(results: dict) -> None:
    """Display benchmark results in table format."""
    table = Table(title="Benchmark Results")

    table.add_column("Workload", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Avg TTFT (ms)", justify="right")
    table.add_column("Avg TBT (ms)", justify="right")
    table.add_column("Output TPS", justify="right")
    table.add_column("Peak Mem (MB)", justify="right")

    for name, metrics in results.items():
        table.add_row(
            name,
            str(metrics.total_requests),
            str(metrics.failed_requests),
            f"{metrics.avg_ttft_ms:.2f}",
            f"{metrics.avg_tbt_ms:.2f}",
            f"{metrics.output_throughput_tps:.2f}",
            str(metrics.peak_mem_mb),
        )

    console.print(table)

    # Display throughput benchmark metrics (aligned with vLLM/SGLang)
    console.print("\n[bold cyan]Throughput Metrics (vLLM/SGLang Compatible)[/bold cyan]")

    for name, metrics in results.items():
        console.print(f"\n[bold]{name}:[/bold]")
        console.print(f"  Output Throughput:   {metrics.output_throughput_tps:>8.2f} tokens/s")
        console.print(f"  Avg Per-Request TPS: {metrics.avg_throughput_tps:>8.2f} tokens/s")
        console.print(f"  Total Throughput:    {metrics.total_throughput_tps:>8.2f} tokens/s")
        console.print(f"  Input Throughput:    {metrics.input_throughput_tps:>8.2f} tokens/s")
        console.print(f"  Request Throughput:  {metrics.request_throughput_rps:>8.2f} req/s")
        console.print(f"  Total Input Tokens:  {metrics.total_input_tokens:>8d}")
        console.print(f"  Total Output Tokens: {metrics.total_output_tokens:>8d}")


def _display_summary_table(data: dict) -> None:
    """Display summary in table format."""
    console.print("\n[bold cyan]Benchmark Summary[/bold cyan]\n")

    # Overall stats
    overall = data["overall"]
    console.print(f"Total workloads: {overall['total_workloads']}")
    console.print(f"Total requests: {overall['total_requests']}")
    console.print(f"Successful: {overall['successful_requests']}")
    console.print(f"Failed: {overall['failed_requests']}")

    # Per-workload table
    table = Table(title="\nWorkload Details")

    table.add_column("Workload", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Avg TTFT (ms)", justify="right")
    table.add_column("Throughput (tok/s)", justify="right")

    for name, metrics in data["workloads"].items():
        table.add_row(
            name,
            str(metrics["total_requests"]),
            str(metrics["failed_requests"]),
            f"{metrics['avg_ttft_ms']:.2f}",
            f"{metrics['avg_throughput_tps']:.2f}",
        )

    console.print(table)


def _display_markdown(data: dict) -> None:
    """Display summary in markdown format."""
    console.print("# Benchmark Results\n")

    overall = data["overall"]
    console.print("## Overall Statistics\n")
    console.print(f"- **Total Workloads**: {overall['total_workloads']}")
    console.print(f"- **Total Requests**: {overall['total_requests']}")
    console.print(f"- **Successful**: {overall['successful_requests']}")
    console.print(f"- **Failed**: {overall['failed_requests']}\n")

    console.print("## Workload Details\n")
    console.print("| Workload | Requests | Errors | Avg TTFT (ms) | Throughput (tok/s) |")
    console.print("|----------|----------|--------|---------------|---------------------|")

    for name, metrics in data["workloads"].items():
        console.print(
            f"| {name} | {metrics['total_requests']} | {metrics['failed_requests']} | "
            f"{metrics['avg_ttft_ms']:.2f} | {metrics['avg_throughput_tps']:.2f} |"
        )


def _normalize_key_part(value: str | int | None) -> str:
    """Normalize one idempotency key part."""
    raw = str(value or "unknown").strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", raw)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or "unknown"


def _extract_workload_for_key(entry: dict) -> str:
    """Extract workload name for idempotency key construction."""
    direct = (
        entry.get("workload", {}).get("name")
        or entry.get("workload_name")
        or entry.get("metadata", {}).get("workload")
    )
    if isinstance(direct, str) and direct.strip():
        return direct.strip().upper()

    notes = str(entry.get("metadata", {}).get("notes") or "")
    q_match = re.search(r"\bQ([1-8])\b", notes, flags=re.IGNORECASE)
    if q_match:
        return f"Q{q_match.group(1)}"

    return "LEGACY"


def _extract_engine_for_key(entry: dict) -> str:
    """Extract engine name for idempotency key construction."""
    raw = entry.get("engine") or entry.get("metadata", {}).get("engine")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    if entry.get("sagellm_version"):
        return "sagellm"
    return "unknown"


def _extract_engine_version_for_key(entry: dict) -> str:
    """Extract engine version for idempotency key construction."""
    raw = (
        entry.get("engine_version")
        or entry.get("metadata", {}).get("engine_version")
        or entry.get("sagellm_version")
    )
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "unknown"


def build_idempotency_key(entry: dict) -> str:
    """Build idempotency key for one leaderboard entry.

    Key dimensions:
    - engine
    - engine version
    - sagellm version
    - workload
    - model name
    - precision
    - hardware model/count
    - node count
    - config type
    """
    cluster = entry.get("cluster") or {}
    parts = [
        _normalize_key_part(_extract_engine_for_key(entry)),
        _normalize_key_part(_extract_engine_version_for_key(entry)),
        _normalize_key_part(entry.get("sagellm_version")),
        _normalize_key_part(_extract_workload_for_key(entry)),
        _normalize_key_part(entry.get("model", {}).get("name")),
        _normalize_key_part(entry.get("model", {}).get("precision")),
        _normalize_key_part(entry.get("hardware", {}).get("chip_model")),
        _normalize_key_part(entry.get("hardware", {}).get("chip_count")),
        _normalize_key_part(cluster.get("node_count", 1)),
        _normalize_key_part(entry.get("config_type")),
    ]
    return "|".join(parts)


def build_canonical_path(entry: dict) -> str:
    """Build canonical dataset path from idempotency key."""
    key = build_idempotency_key(entry)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return f"canonical/{digest}_leaderboard.json"


def _parse_entry_time(entry: dict) -> tuple[datetime | None, datetime | None]:
    """Parse submitted_at and release_date from leaderboard entry metadata."""
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


def _prefer_newer_entry(current: dict, candidate: dict) -> dict:
    """Pick preferred entry between two same-idempotency-key candidates."""
    current_submitted, current_release = _parse_entry_time(current)
    candidate_submitted, candidate_release = _parse_entry_time(candidate)

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


def _normalize_entries_payload(payload: dict | list) -> list[dict]:
    """Normalize leaderboard JSON payload to a list of entries."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


@main.command()
@click.option(
    "--dataset",
    type=str,
    default="intellistream/sagellm-benchmark-results",
    help="Hugging Face dataset repo ID (e.g., intellistream/sagellm-benchmark-results).",
)
@click.option(
    "--input",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="outputs",
    show_default=True,
    help="Input directory to scan recursively for *_leaderboard.json files.",
)
@click.option(
    "--token",
    type=str,
    default=None,
    help="Hugging Face token (fallback to HF_TOKEN env var).",
)
@click.option(
    "--private/--public",
    default=False,
    help="Create dataset repo as private/public if it does not exist.",
)
def upload_hf(dataset: str, input_dir: str, token: str | None, private: bool) -> None:
    """Upload benchmark leaderboard files to Hugging Face dataset."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        console.print("[red]❌ missing dependency: huggingface_hub[/red]")
        console.print("Install with: [cyan]pip install huggingface_hub[/cyan]")
        sys.exit(1)

    resolved_token = token or os.getenv("HF_TOKEN")
    if not resolved_token:
        console.print("[red]❌ HF token not provided[/red]")
        console.print("Use --token or set HF_TOKEN environment variable")
        sys.exit(1)

    hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    os.environ["HF_ENDPOINT"] = hf_endpoint

    input_path = Path(input_dir)
    leaderboard_files = sorted(input_path.rglob("*_leaderboard.json"))

    if not leaderboard_files:
        console.print(f"[red]❌ No leaderboard files found under: {input_path}[/red]")
        sys.exit(1)

    api = HfApi(endpoint=hf_endpoint, token=resolved_token)

    try:
        api.repo_info(repo_id=dataset, repo_type="dataset")
        console.print(f"[green]✓ Dataset exists:[/green] {dataset}")
    except Exception:
        console.print(f"[yellow]⚠ Dataset not found, creating:[/yellow] {dataset}")
        api.create_repo(repo_id=dataset, repo_type="dataset", private=private)
        console.print(f"[green]✓ Created dataset:[/green] {dataset}")

    console.print(f"[cyan]Endpoint:[/cyan] {hf_endpoint}")
    console.print(
        f"[cyan]Scanning[/cyan] {len(leaderboard_files)} leaderboard files from {input_path}"
    )

    canonical_entries: dict[str, dict] = {}
    parse_errors: list[str] = []
    for file_path in leaderboard_files:
        try:
            with open(file_path) as f:
                payload = json.load(f)
        except Exception as exc:
            parse_errors.append(f"{file_path}: {exc}")
            continue

        for entry in _normalize_entries_payload(payload):
            key = build_idempotency_key(entry)
            entry_with_key = json.loads(json.dumps(entry))
            metadata = entry_with_key.setdefault("metadata", {})
            metadata["idempotency_key"] = key
            entry_with_key["canonical_path"] = build_canonical_path(entry_with_key)

            existing = canonical_entries.get(key)
            canonical_entries[key] = (
                _prefer_newer_entry(existing, entry_with_key) if existing else entry_with_key
            )

    if parse_errors:
        console.print("[yellow]⚠ Some files could not be parsed and were skipped:[/yellow]")
        for error in parse_errors:
            console.print(f"  - {error}")

    if not canonical_entries:
        console.print("[red]❌ No valid leaderboard entries found for upload[/red]")
        sys.exit(1)

    console.print(
        f"[cyan]Idempotent entries:[/cyan] {len(canonical_entries)} "
        f"(from {len(leaderboard_files)} files)"
    )

    upload_errors: list[str] = []
    skipped_count = 0
    uploaded_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading canonical entries", total=len(canonical_entries))

        for key, entry in canonical_entries.items():
            path_in_repo = entry["canonical_path"]
            try:
                local_is_newer = True
                try:
                    remote_file = hf_hub_download(
                        repo_id=dataset,
                        filename=path_in_repo,
                        repo_type="dataset",
                        token=resolved_token,
                        endpoint=hf_endpoint,
                    )
                    with open(remote_file) as f:
                        remote_payload = json.load(f)
                    remote_entries = _normalize_entries_payload(remote_payload)
                    if remote_entries:
                        preferred = _prefer_newer_entry(remote_entries[0], entry)
                        local_is_newer = preferred is entry
                except Exception:
                    local_is_newer = True

                if not local_is_newer:
                    skipped_count += 1
                    continue

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", encoding="utf-8", delete=False
                ) as temp_file:
                    json.dump(entry, temp_file, indent=2)
                    temp_path = temp_file.name

                api.upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=path_in_repo,
                    repo_id=dataset,
                    repo_type="dataset",
                    commit_message=(
                        f"Upsert canonical leaderboard {path_in_repo} "
                        f"({datetime.now().isoformat()})"
                    ),
                )
                uploaded_count += 1
                Path(temp_path).unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                upload_errors.append(f"{path_in_repo}: {exc}")
            finally:
                progress.advance(task)

    if upload_errors:
        console.print("[red]❌ Upload completed with errors:[/red]")
        for error in upload_errors:
            console.print(f"  - {error}")
        sys.exit(1)

    console.print("[bold green]✅ Upload complete![/bold green]")
    console.print(f"[green]Uploaded:[/green] {uploaded_count}")
    console.print(f"[yellow]Skipped (remote newer/same):[/yellow] {skipped_count}")
    console.print(f"🔗 https://huggingface.co/datasets/{dataset}")


@main.command()
def aggregate():
    """聚合本地 benchmark 结果并准备上传到 Hugging Face.

    工作流程:
    1. 从 HF 下载最新的公开数据（无需 token）
    2. 扫描本地 outputs/ 目录的新结果
    3. 智能合并（去重，选性能更好的）
    4. 保存到 hf_data/ 目录

    之后用户可以:
        git add hf_data/
        git commit -m "feat: add benchmark results"
        git push
    """
    import subprocess
    from pathlib import Path

    # 找到 aggregate_for_hf.py 脚本
    script_dir = Path(__file__).parent.parent.parent.parent / "scripts"
    aggregate_script = script_dir / "aggregate_for_hf.py"

    if not aggregate_script.exists():
        console.print(f"[red]❌ 未找到聚合脚本: {aggregate_script}[/red]")
        console.print("[yellow]💡 请确保在 sagellm-benchmark 仓库根目录运行[/yellow]")
        sys.exit(1)

    # 运行聚合脚本
    try:
        subprocess.run([sys.executable, str(aggregate_script)], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ 聚合失败: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option(
    "--results",
    default="./benchmark_results",
    show_default=True,
    help="Directory containing JSON benchmark result files.",
)
@click.option(
    "--output",
    default="dashboard.html",
    show_default=True,
    help="Output HTML file path.",
)
@click.option(
    "--title",
    default="SageLLM Performance Leaderboard",
    show_default=True,
    help="Dashboard page title.",
)
@click.option(
    "--sort-by",
    default="throughput_tps",
    type=click.Choice(["throughput_tps", "ttft_ms", "latency_p99_ms", "tbt_ms"]),
    show_default=True,
    help="Default sort column for ranking.",
)
def dashboard(results: str, output: str, title: str, sort_by: str) -> None:
    """生成交互式 HTML 性能排行榜（Dashboard）.

    从 benchmark_results/ 目录加载 JSON 结果，生成可排序的 HTML 排行榜页面，
    支持按场景/数据集分 Tab 展示不同工作负载的性能排名。

    示例:

        sagellm-benchmark dashboard --results ./benchmark_results --output dashboard.html
    """
    from sagellm_benchmark.dashboard import RankingDashboard

    db = RankingDashboard(results_dir=results)
    db.load()

    if not db._entries:
        console.print(f"[yellow]⚠️  No results found in {results}[/yellow]")
        console.print("[dim]Run 'sagellm-benchmark run' to generate results first.[/dim]")
        return

    db.generate(output_path=output, title=title, sort_by=sort_by)
    n = len(db._entries)
    console.print(f"[bold green]✅ Dashboard generated: {output}[/bold green]")
    console.print(f"[green]   Entries: {n} result row(s)[/green]")
    console.print(f"[dim]   Open {output} in a browser to view the leaderboard.[/dim]")


@main.command("workload-template")
@click.option(
    "--output",
    default="workloads_template.json",
    show_default=True,
    help="Output path for the template file (.json or .yaml/.yml).",
)
@click.option(
    "--format",
    "fmt",
    default="json",
    type=click.Choice(["json", "yaml"]),
    show_default=True,
    help="Template file format.",
)
def workload_template(output: str, fmt: str) -> None:
    """生成工作负载配置模板文件 (YAML / JSON).

    生成一个包含预设示例的模板文件，用户可基于此模板自定义 workload 配置，
    然后通过 ``--workload-file`` 参数加载。

    示例:

        sagellm-benchmark workload-template --output my_workloads.yaml --format yaml

        sagellm-benchmark run --workload-file my_workloads.yaml ...
    """
    from sagellm_benchmark.workloads import WorkloadTemplateGenerator

    if fmt == "yaml":
        WorkloadTemplateGenerator.generate_yaml(output)
    else:
        WorkloadTemplateGenerator.generate_json(output)

    console.print(f"[bold green]✅ Workload template written to: {output}[/bold green]")
    console.print("[dim]Edit the file and use --workload-file to load custom workloads.[/dim]")


if __name__ == "__main__":
    main()
