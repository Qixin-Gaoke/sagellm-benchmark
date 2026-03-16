"""Shared compare runner utilities for OpenAI-compatible endpoint benchmarks."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING, Any

from sagellm_benchmark.clients.openai_client import GatewayClient
from sagellm_benchmark.datasets import build_serving_requests
from sagellm_benchmark.workload_profiles import WorkloadScenarioPlan

if TYPE_CHECKING:
    from sagellm_benchmark.types import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompareScenario:
    """One benchmark scenario for compare transports."""

    name: str
    batch_size: int
    prompt: str
    prompt_tokens: int
    output_tokens: int
    scenario_source: str
    workload_profile: str
    supplements: tuple[str, ...]
    dataset_name: str


@dataclass(frozen=True)
class CompareRequestResult:
    """Normalized request result across compare transports."""

    ok: bool
    status_code: int | None
    elapsed_ms: float
    error: str | None = None
    completion_text: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0
    tbt_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    throughput_tps: float = 0.0
    itl_list: list[float] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None


def synthetic_prompt(prompt_tokens: int, filler_word: str = "benchmark") -> str:
    """Build a deterministic synthetic prompt of roughly the requested token length."""
    words_needed = max(10, int(prompt_tokens / 1.3))
    return " ".join([filler_word] * words_needed)


def percentile(values: list[float], p: float) -> float:
    """Compute a percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = index - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def result_from_benchmark_result(result: BenchmarkResult) -> CompareRequestResult:
    """Convert BenchmarkResult into a transport-agnostic compare result."""
    if result.success and result.metrics is not None:
        prompt_tokens = result.prompt_tokens
        output_tokens = result.output_tokens
        return CompareRequestResult(
            ok=True,
            status_code=200,
            elapsed_ms=result.e2e_latency_ms or 0.0,
            completion_text=result.output_text,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            ttft_ms=result.metrics.ttft_ms,
            tbt_ms=result.metrics.tbt_ms,
            tpot_ms=result.metrics.tpot_ms,
            e2e_latency_ms=result.e2e_latency_ms,
            throughput_tps=result.metrics.throughput_tps,
            itl_list=list(result.itl_list),
        )

    return CompareRequestResult(
        ok=False,
        status_code=None,
        elapsed_ms=result.e2e_latency_ms or 0.0,
        error=result.error,
    )


def result_from_nonstream_mapping(result: dict[str, object]) -> CompareRequestResult:
    """Convert legacy non-stream compare result mapping into CompareRequestResult."""
    prompt_tokens = int(result.get("prompt_tokens") or 0)
    output_tokens = int(result.get("completion_tokens") or 0)
    elapsed_ms = float(result.get("elapsed_ms") or 0.0)
    ok = bool(result.get("ok"))
    return CompareRequestResult(
        ok=ok,
        status_code=int(result["status_code"]) if result.get("status_code") is not None else None,
        elapsed_ms=elapsed_ms,
        error=str(result.get("error") or "") or None,
        completion_text=str(result.get("completion_text") or ""),
        finish_reason=(str(result.get("finish_reason")) if result.get("finish_reason") else None),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=int(result.get("total_tokens") or (prompt_tokens + output_tokens)),
        ttft_ms=elapsed_ms,
        tbt_ms=0.0,
        tpot_ms=(elapsed_ms / output_tokens) if output_tokens > 0 else 0.0,
        e2e_latency_ms=elapsed_ms,
        throughput_tps=(output_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
        raw_response=result.get("raw_response")
        if isinstance(result.get("raw_response"), dict)
        else None,
    )


def summarize_nonstream_batch(
    *,
    batch_size: int,
    round_index: int,
    request_results: list[CompareRequestResult],
    wall_time_ms: float,
) -> dict[str, object]:
    """Summarize one non-stream batch while preserving existing output shape."""
    successes = [result for result in request_results if result.ok]
    failures = [result for result in request_results if not result.ok]
    avg_latency_ms = (
        mean(result.elapsed_ms for result in request_results) if request_results else 0.0
    )
    throughput_rps = (len(successes) * 1000.0 / wall_time_ms) if wall_time_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "round_index": round_index,
        "request_count": len(request_results),
        "success_count": len(successes),
        "error_count": len(failures),
        "wall_time_ms": wall_time_ms,
        "avg_request_latency_ms": avg_latency_ms,
        "throughput_rps": throughput_rps,
        "prompt_tokens": sum(result.prompt_tokens for result in successes),
        "completion_tokens": sum(result.output_tokens for result in successes),
        "sample_output": successes[0].completion_text if successes else "",
        "errors": [str(result.error or "") for result in failures],
        "requests": [
            {
                "ok": result.ok,
                "status_code": result.status_code,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
                "completion_text": result.completion_text,
                "finish_reason": result.finish_reason,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "ttft_ms": result.ttft_ms,
                "tbt_ms": result.tbt_ms,
                "tpot_ms": result.tpot_ms,
                "e2e_latency_ms": result.e2e_latency_ms,
                "throughput_tps": result.throughput_tps,
                "raw_response": result.raw_response,
            }
            for result in request_results
        ],
    }


def summarize_compare_row(
    *,
    requested_model: str,
    effective_model: str,
    scenario: CompareScenario,
    request_results: list[CompareRequestResult],
    wall_time_s: float,
    mode: str,
    transport: str,
) -> dict[str, Any]:
    """Summarize one compare scenario into the canonical row shape."""
    successful = [result for result in request_results if result.ok]
    failed = [result for result in request_results if not result.ok]
    ttft_values = [result.ttft_ms for result in successful if result.ttft_ms > 0]
    tbt_values = [result.tbt_ms for result in successful if result.tbt_ms > 0]
    tpot_values = [result.tpot_ms for result in successful if result.tpot_ms > 0]
    itl_values = [sample for result in successful for sample in result.itl_list if sample > 0]
    e2e_values = [result.e2e_latency_ms for result in successful if result.e2e_latency_ms > 0]
    throughput_values = [
        result.throughput_tps for result in successful if result.throughput_tps > 0
    ]
    total_output_tokens = sum(result.output_tokens for result in successful)
    output_throughput_tps = (
        total_output_tokens / wall_time_s if successful and wall_time_s > 0 else 0.0
    )
    request_throughput_rps = (
        len(successful) / wall_time_s if successful and wall_time_s > 0 else 0.0
    )

    return {
        "model": requested_model,
        "effective_model": effective_model,
        "precision": "live",
        "scenario": scenario.name,
        "batch_size": scenario.batch_size,
        "ttft_ms": mean(ttft_values) if ttft_values else 0.0,
        "tbt_ms": mean(tbt_values) if tbt_values else 0.0,
        "tpot_ms": mean(tpot_values) if tpot_values else 0.0,
        "avg_itl_ms": mean(itl_values) if itl_values else 0.0,
        "p50_itl_ms": percentile(itl_values, 50),
        "p95_itl_ms": percentile(itl_values, 95),
        "p99_itl_ms": percentile(itl_values, 99),
        "throughput_tps": mean(throughput_values) if throughput_values else 0.0,
        "output_throughput_tps": output_throughput_tps,
        "request_throughput_rps": request_throughput_rps,
        "latency_p50_ms": percentile(e2e_values, 50),
        "latency_p95_ms": percentile(e2e_values, 95),
        "latency_p99_ms": percentile(e2e_values, 99),
        "avg_e2el_ms": mean(e2e_values) if e2e_values else 0.0,
        "memory_mb": 0.0,
        "mode": mode,
        "transport": transport,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "scenario_source": scenario.scenario_source,
        "workload_profile": scenario.workload_profile,
        "supplements": list(scenario.supplements),
        "dataset_name": scenario.dataset_name,
    }


def summarize_compare_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize compare rows for target-level output."""
    avg_itl_values = [
        float(row.get("avg_itl_ms", 0.0)) for row in rows if float(row.get("avg_itl_ms", 0.0)) > 0
    ]
    p50_itl_values = [
        float(row.get("p50_itl_ms", 0.0)) for row in rows if float(row.get("p50_itl_ms", 0.0)) > 0
    ]
    p95_itl_values = [
        float(row.get("p95_itl_ms", 0.0)) for row in rows if float(row.get("p95_itl_ms", 0.0)) > 0
    ]
    p99_itl_values = [
        float(row.get("p99_itl_ms", 0.0)) for row in rows if float(row.get("p99_itl_ms", 0.0)) > 0
    ]
    return {
        "total_rows": len(rows),
        "avg_ttft_ms": mean(row["ttft_ms"] for row in rows) if rows else 0.0,
        "avg_tbt_ms": mean(row["tbt_ms"] for row in rows) if rows else 0.0,
        "avg_tpot_ms": mean(float(row.get("tpot_ms", 0.0)) for row in rows) if rows else 0.0,
        "avg_itl_ms": mean(avg_itl_values) if avg_itl_values else 0.0,
        "p50_itl_ms": mean(p50_itl_values) if p50_itl_values else 0.0,
        "p95_itl_ms": mean(p95_itl_values) if p95_itl_values else 0.0,
        "p99_itl_ms": mean(p99_itl_values) if p99_itl_values else 0.0,
        "avg_e2el_ms": mean(float(row.get("avg_e2el_ms", 0.0)) for row in rows) if rows else 0.0,
        "avg_throughput_tps": mean(row["throughput_tps"] for row in rows) if rows else 0.0,
        "avg_output_throughput_tps": (
            mean(float(row.get("output_throughput_tps", row["throughput_tps"])) for row in rows)
            if rows
            else 0.0
        ),
        "avg_request_throughput_rps": (
            mean(float(row.get("request_throughput_rps", 0.0)) for row in rows) if rows else 0.0
        ),
    }


async def wait_for_endpoint_ready(
    client: GatewayClient,
    *,
    backend_url: str,
    server_wait_s: float,
) -> None:
    """Wait until an endpoint is healthy or the deadline expires."""
    deadline = time.monotonic() + server_wait_s
    ready = False
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        healthy = await client.health_check(timeout=5.0)
        if healthy:
            ready = True
            logger.info("Server ready after %s probe(s)", attempt)
            break
        wait = min(3.0, deadline - time.monotonic())
        if wait > 0:
            logger.info(
                "Server %s not ready yet (attempt %s); retrying in %.1fs...",
                backend_url,
                attempt,
                wait,
            )
            await asyncio.sleep(wait)

    if not ready:
        logger.error(
            "Server at %s did not become ready within %.0fs. Proceeding anyway.",
            backend_url,
            server_wait_s,
        )


async def _run_stream_compare_target_async(
    *,
    model: str,
    backend_url: str,
    batch_sizes: tuple[int, ...],
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
    max_seq_len_override: int | None,
    max_output_tokens_override: int | None,
    scenarios: tuple[WorkloadScenarioPlan, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the live stream compare transport and return canonical rows + summary."""
    from sagellm_benchmark.performance.model_benchmarks import _discover_max_seq_len
    from sagellm_benchmark.types import BenchmarkRequest

    if not scenarios:
        raise ValueError("compare scenarios are required")

    client = GatewayClient(
        base_url=backend_url,
        api_key=api_key,
        timeout=request_timeout,
        endpoint_type="chat",
    )

    rows: list[dict[str, Any]] = []
    try:
        await wait_for_endpoint_ready(client, backend_url=backend_url, server_wait_s=server_wait_s)
        discovered_model = await client.discover_model(timeout=5.0)
        effective_model = discovered_model or model

        if discovered_model and discovered_model != model:
            logger.warning(
                "Requested model '%s' does not match server model '%s'. Using server model.",
                model,
                discovered_model,
            )

        if max_seq_len_override is not None:
            max_seq_len = max_seq_len_override
        else:
            max_seq_len = await _discover_max_seq_len(
                client=client,
                model_path=effective_model,
                backend_url=backend_url,
            )

        for scenario in scenarios:
            effective_output_tokens = scenario.output_len
            if (
                max_output_tokens_override is not None
                and effective_output_tokens > max_output_tokens_override
            ):
                effective_output_tokens = max_output_tokens_override
            effective_prompt_tokens = min(
                scenario.input_len,
                max(10, max_seq_len - effective_output_tokens - 10),
            )

            if scenario.dataset_name == "random":
                prompt = synthetic_prompt(effective_prompt_tokens)
                requests = [
                    BenchmarkRequest(
                        prompt=prompt,
                        max_tokens=effective_output_tokens,
                        request_id=(f"live-{scenario.scenario_name}-{scenario.batch_size}-{index}"),
                        model=effective_model,
                        stream=True,
                    )
                    for index in range(scenario.batch_size)
                ]
                started_at = time.perf_counter()
                results = await client.generate_batch(requests, concurrent=True)
                wall_time_s = max(time.perf_counter() - started_at, 1e-9)
                compare_results = [result_from_benchmark_result(result) for result in results]
            else:
                serving_dataset_name = "sharegpt"
                sharegpt_path = scenario.dataset_path if scenario.dataset_name == "custom" else None
                dataset_requests = build_serving_requests(
                    dataset_name=serving_dataset_name,
                    num_prompts=scenario.num_prompts,
                    input_len=effective_prompt_tokens,
                    output_len=effective_output_tokens,
                    model=effective_model,
                    stream=True,
                    seed=0,
                    sharegpt_path=sharegpt_path,
                )

                compare_results = []
                wall_time_s = 0.0
                for batch_start in range(0, len(dataset_requests), scenario.batch_size):
                    batch = dataset_requests[batch_start : batch_start + scenario.batch_size]
                    requests = [
                        BenchmarkRequest(
                            prompt=request.prompt,
                            max_tokens=request.max_tokens,
                            request_id=(
                                f"live-{scenario.scenario_name}-"
                                f"b{scenario.batch_size}-{batch_start + index}"
                            ),
                            model=effective_model,
                            stream=True,
                        )
                        for index, request in enumerate(batch)
                    ]
                    started_at = time.perf_counter()
                    results = await client.generate_batch(requests, concurrent=True)
                    wall_time_s += max(time.perf_counter() - started_at, 1e-9)
                    compare_results.extend(
                        result_from_benchmark_result(result) for result in results
                    )

                prompt = f"dataset:{scenario.dataset_name}"

            rows.append(
                summarize_compare_row(
                    requested_model=model,
                    effective_model=effective_model,
                    scenario=CompareScenario(
                        name=scenario.scenario_name,
                        batch_size=scenario.batch_size,
                        prompt=prompt,
                        prompt_tokens=effective_prompt_tokens,
                        output_tokens=effective_output_tokens,
                        scenario_source=scenario.scenario_source,
                        workload_profile=scenario.workload_profile,
                        supplements=scenario.supplements,
                        dataset_name=scenario.dataset_name,
                    ),
                    request_results=compare_results,
                    wall_time_s=max(wall_time_s, 1e-9),
                    mode="live",
                    transport="stream",
                )
            )
    finally:
        await client.close()

    return rows, summarize_compare_rows(rows)


def run_stream_compare_target(
    *,
    model: str,
    backend_url: str,
    batch_sizes: tuple[int, ...],
    api_key: str,
    request_timeout: float,
    server_wait_s: float,
    max_seq_len_override: int | None,
    max_output_tokens_override: int | None,
    scenarios: tuple[WorkloadScenarioPlan, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Synchronous wrapper for the shared stream compare transport."""
    return asyncio.run(
        _run_stream_compare_target_async(
            model=model,
            backend_url=backend_url,
            batch_sizes=batch_sizes,
            api_key=api_key,
            request_timeout=request_timeout,
            server_wait_s=server_wait_s,
            max_seq_len_override=max_seq_len_override,
            max_output_tokens_override=max_output_tokens_override,
            scenarios=scenarios,
        )
    )


async def run_nonstream_batch(
    *,
    target: Any,
    request_factory: Callable[[int], Any],
    batch_size: int,
    request_fn: Callable[[Any, Any], dict[str, object]],
) -> tuple[list[CompareRequestResult], float]:
    """Run one non-stream batch through the shared async compare runner."""

    async def _run_one(index: int) -> CompareRequestResult:
        request_config = request_factory(index)
        result = await asyncio.to_thread(request_fn, target, request_config)
        return result_from_nonstream_mapping(result)

    started_at = time.perf_counter()
    request_results = await asyncio.gather(*(_run_one(index) for index in range(batch_size)))
    wall_time_ms = (time.perf_counter() - started_at) * 1000.0
    return list(request_results), wall_time_ms
