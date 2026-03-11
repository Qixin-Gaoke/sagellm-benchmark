"""Reusable non-stream compare runner for OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter


def _slugify_filename(value: str) -> str:
    """Convert a label to a filesystem-safe filename stem."""
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "target"


def _create_output_dir(output_dir: str | None, prefix: str = "nonstream_compare") -> Path:
    """Create the output directory for non-stream compare runs."""
    return (
        Path(output_dir)
        if output_dir
        else Path("benchmark_results") / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def _normalize_chat_completions_url(url: str) -> str:
    """Normalize an endpoint URL to the chat completions path."""
    normalized = url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


@dataclass(frozen=True)
class NonStreamTarget:
    """Target endpoint participating in the compare run."""

    label: str
    url: str


@dataclass(frozen=True)
class NonStreamRequestConfig:
    """Per-request settings shared by all targets."""

    model: str
    prompt: str
    max_tokens: int
    temperature: float
    api_key: str
    request_timeout: float


@dataclass(frozen=True)
class NonStreamCompareConfig:
    """Configuration for a non-stream compare run."""

    targets: tuple[NonStreamTarget, ...]
    model: str
    prompt: str
    batch_sizes: tuple[int, ...]
    warmup_rounds: int
    rounds: int
    max_tokens: int
    temperature: float
    api_key: str
    request_timeout: float
    output_dir: str | None = None


def parse_target_spec(spec: str) -> NonStreamTarget:
    """Parse a LABEL=URL target specification."""
    if "=" not in spec:
        raise ValueError(
            f"Invalid target '{spec}'. Expected LABEL=URL, for example sagellm=http://127.0.0.1:8901/v1"
        )

    label, url = spec.split("=", 1)
    label = label.strip()
    url = url.strip()
    if not label or not url:
        raise ValueError(
            f"Invalid target '{spec}'. Both label and URL are required in LABEL=URL format."
        )
    return NonStreamTarget(label=label, url=url)


def _build_request_payload(config: NonStreamRequestConfig) -> bytes:
    """Create the OpenAI-compatible non-stream request payload."""
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": config.prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": False,
    }
    return json.dumps(payload).encode("utf-8")


def send_nonstream_request(
    target: NonStreamTarget,
    request_config: NonStreamRequestConfig,
) -> dict[str, object]:
    """Send a single non-stream chat completion request."""
    request = urllib.request.Request(
        _normalize_chat_completions_url(target.url),
        data=_build_request_payload(request_config),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {request_config.api_key}",
        },
        method="POST",
    )

    started_at = perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=request_config.request_timeout) as response:
            body = response.read().decode("utf-8")
            payload = json.loads(body)
            elapsed_ms = (perf_counter() - started_at) * 1000.0
    except urllib.error.HTTPError as exc:
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        error_body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status_code": exc.code,
            "elapsed_ms": elapsed_ms,
            "error": f"HTTP {exc.code}: {error_body}",
        }
    except urllib.error.URLError as exc:
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        return {
            "ok": False,
            "status_code": None,
            "elapsed_ms": elapsed_ms,
            "error": str(exc.reason),
        }
    except Exception as exc:  # pragma: no cover - safety net for unexpected transport errors
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        return {
            "ok": False,
            "status_code": None,
            "elapsed_ms": elapsed_ms,
            "error": str(exc),
        }

    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    return {
        "ok": True,
        "status_code": 200,
        "elapsed_ms": elapsed_ms,
        "completion_text": message.get("content") or choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "raw_response": payload,
    }


def _summarize_batch(
    *,
    batch_size: int,
    round_index: int,
    request_results: list[dict[str, object]],
    wall_time_ms: float,
) -> dict[str, object]:
    """Summarize one concurrent batch execution."""
    successes = [result for result in request_results if result["ok"]]
    failures = [result for result in request_results if not result["ok"]]

    avg_latency_ms = (
        sum(float(result["elapsed_ms"]) for result in request_results) / len(request_results)
        if request_results
        else 0.0
    )
    throughput_rps = (len(successes) * 1000.0 / wall_time_ms) if wall_time_ms > 0 else 0.0

    first_success = successes[0] if successes else None
    return {
        "batch_size": batch_size,
        "round_index": round_index,
        "request_count": len(request_results),
        "success_count": len(successes),
        "error_count": len(failures),
        "wall_time_ms": wall_time_ms,
        "avg_request_latency_ms": avg_latency_ms,
        "throughput_rps": throughput_rps,
        "prompt_tokens": sum(int(result.get("prompt_tokens") or 0) for result in successes),
        "completion_tokens": sum(int(result.get("completion_tokens") or 0) for result in successes),
        "sample_output": first_success.get("completion_text", "") if first_success else "",
        "errors": [str(result.get("error") or "") for result in failures],
        "requests": request_results,
    }


def _run_batch(
    *,
    target: NonStreamTarget,
    request_config: NonStreamRequestConfig,
    batch_size: int,
    round_index: int,
    request_fn: callable,
) -> dict[str, object]:
    """Run one concurrent batch against a single endpoint."""
    started_at = perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(request_fn, target, request_config) for _ in range(batch_size)]
        request_results = [future.result() for future in futures]
    wall_time_ms = (perf_counter() - started_at) * 1000.0
    return _summarize_batch(
        batch_size=batch_size,
        round_index=round_index,
        request_results=request_results,
        wall_time_ms=wall_time_ms,
    )


def _summarize_target(
    target: NonStreamTarget,
    batch_results: list[dict[str, object]],
    warmup_results: list[dict[str, object]],
) -> dict[str, object]:
    """Build per-target summary metrics."""
    request_results = [
        request for batch in batch_results for request in batch["requests"] if bool(request["ok"])
    ]
    total_requests = sum(int(batch["request_count"]) for batch in batch_results)
    total_success = len(request_results)
    total_errors = total_requests - total_success
    total_wall_time_ms = sum(float(batch["wall_time_ms"]) for batch in batch_results)

    avg_latency_ms = (
        sum(float(request["elapsed_ms"]) for request in request_results) / total_success
        if total_success
        else 0.0
    )
    avg_prompt_tokens = (
        sum(int(request.get("prompt_tokens") or 0) for request in request_results) / total_success
        if total_success
        else 0.0
    )
    avg_completion_tokens = (
        sum(int(request.get("completion_tokens") or 0) for request in request_results)
        / total_success
        if total_success
        else 0.0
    )
    throughput_rps = (
        (total_success * 1000.0 / total_wall_time_ms) if total_wall_time_ms > 0 else 0.0
    )

    return {
        "label": target.label,
        "url": target.url,
        "summary": {
            "total_batches": len(batch_results),
            "warmup_rounds": len(warmup_results),
            "total_requests": total_requests,
            "successful_requests": total_success,
            "failed_requests": total_errors,
            "avg_latency_ms": avg_latency_ms,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "throughput_rps": throughput_rps,
        },
        "warmups": warmup_results,
        "batches": batch_results,
    }


def _build_comparison_summary(target_results: list[dict[str, object]]) -> dict[str, object]:
    """Build a baseline-relative comparison summary."""
    if not target_results:
        return {"baseline": None, "targets": []}

    baseline = target_results[0]
    baseline_summary = baseline["summary"]
    baseline_latency = float(baseline_summary["avg_latency_ms"])
    baseline_throughput = float(baseline_summary["throughput_rps"])
    baseline_completion_tokens = float(baseline_summary["avg_completion_tokens"])

    targets: list[dict[str, object]] = []
    for target in target_results:
        summary = target["summary"]
        targets.append(
            {
                "label": target["label"],
                "url": target["url"],
                **summary,
                "delta_vs_baseline": {
                    "avg_latency_ms": float(summary["avg_latency_ms"]) - baseline_latency,
                    "throughput_rps": float(summary["throughput_rps"]) - baseline_throughput,
                    "avg_completion_tokens": (
                        float(summary["avg_completion_tokens"]) - baseline_completion_tokens
                    ),
                },
            }
        )

    return {"baseline": baseline["label"], "targets": targets}


def _format_markdown(compare_result: dict[str, object]) -> str:
    """Render a concise markdown report."""
    lines = [
        "# Non-Stream Endpoint Comparison Report",
        "",
        f"- Model: {compare_result['model']}",
        f"- Prompt: {compare_result['prompt']}",
        f"- Baseline: {compare_result['baseline']}",
        f"- Batch sizes: {', '.join(str(size) for size in compare_result['batch_sizes'])}",
        "",
        "| Target | Success / Total | Avg Latency (ms) | Throughput (req/s) | Avg Completion Tokens | Delta Latency | Delta Throughput |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for target in compare_result["targets"]:
        delta = target["delta_vs_baseline"]
        lines.append(
            "| {label} | {ok}/{total} | {latency:.2f} | {throughput:.2f} | {completion:.2f} | {d_latency:+.2f} | {d_throughput:+.2f} |".format(
                label=target["label"],
                ok=int(target["successful_requests"]),
                total=int(target["total_requests"]),
                latency=float(target["avg_latency_ms"]),
                throughput=float(target["throughput_rps"]),
                completion=float(target["avg_completion_tokens"]),
                d_latency=float(delta["avg_latency_ms"]),
                d_throughput=float(delta["throughput_rps"]),
            )
        )

    lines.extend(["", "## Targets", ""])
    for target in compare_result["targets"]:
        lines.append(f"- {target['label']}: {target['url']}")

    return "\n".join(lines)


def run_nonstream_compare(
    config: NonStreamCompareConfig,
    request_fn=send_nonstream_request,
) -> Path:
    """Run the non-stream compare benchmark and write artifacts."""
    if len(config.targets) < 2:
        raise ValueError("Repeat --target at least twice to compare multiple endpoints.")

    output_dir = _create_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_config = NonStreamRequestConfig(
        model=config.model,
        prompt=config.prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        api_key=config.api_key,
        request_timeout=config.request_timeout,
    )

    target_results: list[dict[str, object]] = []
    for target in config.targets:
        warmup_results = [request_fn(target, request_config) for _ in range(config.warmup_rounds)]
        batch_results: list[dict[str, object]] = []
        for batch_size in config.batch_sizes:
            for round_index in range(1, config.rounds + 1):
                batch_results.append(
                    _run_batch(
                        target=target,
                        request_config=request_config,
                        batch_size=batch_size,
                        round_index=round_index,
                        request_fn=request_fn,
                    )
                )

        target_result = _summarize_target(target, batch_results, warmup_results)
        target_results.append(target_result)

        target_path = output_dir / f"{_slugify_filename(target.label)}.json"
        target_path.write_text(
            json.dumps(
                {
                    "kind": "nonstream_compare_target",
                    "model": config.model,
                    "prompt": config.prompt,
                    "max_tokens": config.max_tokens,
                    **target_result,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    summary = _build_comparison_summary(target_results)
    compare_result = {
        "kind": "nonstream_compare",
        "model": config.model,
        "prompt": config.prompt,
        "batch_sizes": list(config.batch_sizes),
        **summary,
    }

    (output_dir / "comparison.json").write_text(
        json.dumps(compare_result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(
        _format_markdown(compare_result) + "\n",
        encoding="utf-8",
    )

    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the standalone script argument parser."""
    parser = argparse.ArgumentParser(
        description="Run reusable non-stream chat completion comparisons across endpoints.",
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help=(
            "Comparison target in LABEL=URL format. Repeat to compare multiple OpenAI-compatible "
            "endpoints."
        ),
    )
    parser.add_argument("--model", required=True, help="Requested model name for the compare run.")
    parser.add_argument(
        "--prompt",
        default="请用一句话介绍你自己。",
        help="Prompt sent to each endpoint.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_sizes",
        type=int,
        action="append",
        default=None,
        help="Batch size to benchmark. Repeat for multiple values.",
    )
    parser.add_argument("--warmup-rounds", type=int, default=1, help="Warmup requests per target.")
    parser.add_argument("--rounds", type=int, default=1, help="Measured rounds per batch size.")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--api-key",
        default="sagellm-benchmark",
        help="API key used for all targets.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save artifacts. Defaults to benchmark_results/nonstream_compare_<timestamp>.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Standalone entrypoint for direct script execution."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = NonStreamCompareConfig(
        targets=tuple(parse_target_spec(spec) for spec in args.target),
        model=args.model,
        prompt=args.prompt,
        batch_sizes=tuple(args.batch_sizes or [1, 2]),
        warmup_rounds=args.warmup_rounds,
        rounds=args.rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        api_key=args.api_key,
        request_timeout=args.request_timeout,
        output_dir=args.output_dir,
    )
    output_dir = run_nonstream_compare(config)
    print(f"Non-stream compare artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
