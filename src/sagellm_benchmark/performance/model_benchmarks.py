"""E2E model-level benchmarks for sagellm-benchmark (#45/#46)."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from hashlib import sha256
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt_tokens: int
    output_tokens: int
    batch_size: int


def run_e2e_model_benchmarks(
    *,
    models: list[str],
    batch_sizes: list[int],
    precisions: list[str] | None = None,
    simulate: bool = True,
    backend_url: str = "http://localhost:8000/v1",
    api_key: str = "sagellm-benchmark",
    request_timeout: float = 120.0,
    server_wait_s: float = 30.0,
    max_seq_len: int | None = None,
    max_output_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Run E2E model benchmarks.

    Uses deterministic simulation mode by default to keep CI stable.
    In live mode (simulate=False), sends real requests to an OpenAI-compatible
    API server (e.g., sagellm-gateway or sagellm-core engine_server).

    Args:
        models: List of model names/paths to benchmark.
        batch_sizes: List of batch sizes to test.
        precisions: Precision labels (simulation only; live mode uses server precision).
        simulate: If True, use deterministic simulation. If False, run live requests.
        backend_url: API base URL for live mode (default: http://localhost:8000/v1).
        api_key: API key for live mode (default: sagellm-benchmark).
        request_timeout: Per-request timeout in seconds for live mode.
        server_wait_s: Max seconds to wait for server to become ready in live mode.
        max_seq_len: Maximum sequence length (prompt + output) the model supports.
            If None, auto-detected from the server; falls back to 1024. Used in
            live mode to clamp prompts so they never exceed the model's context window.
        max_output_tokens: Hard cap on output tokens per request in live mode.
            Useful for CPU/slow models where the full scenario output length would exceed
            the request timeout. E.g. ``--max-output-tokens 16`` for tiny CPU models.

    Returns:
        List of result rows with benchmark metrics.
    """
    scenarios: list[Scenario] = []
    for batch_size in batch_sizes:
        scenarios.extend(
            [
                Scenario(f"short_b{batch_size}", 128, 128, batch_size),
                Scenario(f"long_b{batch_size}", 2048, 512, batch_size),
            ]
        )

    rows: list[dict[str, Any]] = []

    if simulate:
        precision_values = precisions or ["fp16", "int8"]
        for model in models:
            for precision in precision_values:
                for scenario in scenarios:
                    seed = _stable_seed(model, precision, scenario.name)
                    rng = random.Random(seed)

                    model_factor = 1.2 if "Llama" in model else (0.9 if "Phi" in model else 1.0)
                    context_factor = 1.8 if scenario.prompt_tokens > 1000 else 1.0
                    precision_factor = (
                        0.85 if precision == "int8" else (1.15 if precision == "fp32" else 1.0)
                    )

                    ttft = (
                        45.0
                        * model_factor
                        * context_factor
                        * precision_factor
                        * (1 + rng.uniform(-0.08, 0.08))
                    )
                    tbt = (
                        9.0
                        * model_factor
                        * context_factor
                        * precision_factor
                        * (1 + rng.uniform(-0.08, 0.08))
                    )

                    throughput = 100.0 / model_factor / context_factor / precision_factor
                    throughput *= scenario.batch_size**0.4
                    throughput *= 1 + rng.uniform(-0.08, 0.08)

                    latencies: list[float] = []
                    for _ in range(max(3, scenario.batch_size)):
                        latencies.append(
                            ttft
                            + tbt
                            * max(1, scenario.output_tokens - 1)
                            / max(1.0, scenario.batch_size * 0.7)
                        )

                    rows.append(
                        {
                            "model": model,
                            "precision": precision,
                            "scenario": scenario.name,
                            "batch_size": scenario.batch_size,
                            "ttft_ms": ttft,
                            "tbt_ms": tbt,
                            "throughput_tps": throughput,
                            "output_throughput_tps": throughput * scenario.batch_size,
                            "latency_p50_ms": _percentile(latencies, 50),
                            "latency_p95_ms": _percentile(latencies, 95),
                            "latency_p99_ms": _percentile(latencies, 99),
                            "memory_mb": 5000.0 * model_factor + scenario.prompt_tokens * 0.2,
                            "mode": "simulate",
                        }
                    )
    else:
        # Live mode: send real requests to an OpenAI-compatible API server
        rows = asyncio.run(
            _run_live_benchmarks(
                models=models,
                scenarios=scenarios,
                backend_url=backend_url,
                api_key=api_key,
                request_timeout=request_timeout,
                server_wait_s=server_wait_s,
                max_seq_len_override=max_seq_len,
                max_output_tokens_override=max_output_tokens,
            )
        )

    return rows


async def _discover_max_seq_len(
    client: Any,
    model_path: str,
    backend_url: str,
) -> int:
    """Discover the maximum sequence length for a model.

    Tries in order:
    1. GET /info from the engine server (checks 'max_model_len' / 'max_position_embeddings')
    2. transformers AutoConfig.from_pretrained(model_path) → max_position_embeddings
    3. Hard fallback: 1024

    Args:
        client: GatewayClient (used only for the /info probe).
        model_path: Model path or HuggingFace repo ID.
        backend_url: API base URL (used to derive server root for /info).

    Returns:
        Maximum sequence length as int.
    """
    _fallback_seq_len = 1024

    # 1. Query /info (sagellm engine_server)
    base = backend_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.get(f"{root}/info")
        if r.status_code == 200:
            data = r.json()
            for key in ("max_model_len", "max_position_embeddings", "max_seq_len", "n_positions"):
                val = data.get(key)
                if isinstance(val, int) and val > 0:
                    logger.info(f"Discovered max_seq_len={val} from /info ({key})")
                    return val
    except Exception as e:
        logger.debug(f"/info max_seq_len probe failed: {e}")

    # 2. transformers AutoConfig (works for local paths and cached HF models)
    try:
        from transformers import AutoConfig  # type: ignore[import-untyped]

        from sagellm_benchmark.clients.openai_client import GatewayClient

        config_source, local_only = GatewayClient._resolve_tokenizer_source(model_path)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        cfg = AutoConfig.from_pretrained(
            config_source,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                logger.info(
                    "Discovered max_seq_len=%s from AutoConfig.%s source=%s local_only=%s",
                    val,
                    attr,
                    config_source,
                    local_only,
                )
                return val
    except Exception as e:
        logger.debug(f"AutoConfig max_seq_len probe failed: {e}")

    logger.warning(
        f"Could not determine max_seq_len for '{model_path}'; defaulting to {_fallback_seq_len}. "
        "Override with --max-seq-len if your model supports a longer context."
    )
    return _fallback_seq_len


async def _run_live_benchmarks(
    *,
    models: list[str],
    scenarios: list[Scenario],
    backend_url: str,
    api_key: str,
    request_timeout: float,
    server_wait_s: float = 30.0,
    max_seq_len_override: int | None = None,
    max_output_tokens_override: int | None = None,
) -> list[dict[str, Any]]:
    """Run live E2E benchmarks against a real API server.

    Args:
        models: Model names/paths to benchmark.
        scenarios: Benchmark scenarios.
        backend_url: API base URL.
        api_key: API key.
        request_timeout: Per-request timeout (seconds).
        server_wait_s: Max seconds to wait for server to become ready.
        max_seq_len_override: If set, skip auto-detection and use this value.
        max_output_tokens_override: Hard cap on output tokens per request.

    Returns:
        List of result rows.
    """
    import time

    from sagellm_benchmark.clients.openai_client import GatewayClient

    rows: list[dict[str, Any]] = []

    # Single shared client for all models (server serves one model at a time)
    client = GatewayClient(
        base_url=backend_url,
        api_key=api_key,
        timeout=request_timeout,
    )

    # --- Pre-flight: wait for server to become ready ---
    logger.info(f"Waiting for server at {backend_url} (up to {server_wait_s:.0f}s)...")
    deadline = time.monotonic() + server_wait_s
    ready = False
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        healthy = await client.health_check(timeout=5.0)
        if healthy:
            ready = True
            logger.info(f"Server ready after {attempt} probe(s)")
            break
        wait = min(3.0, deadline - time.monotonic())
        if wait > 0:
            logger.info(f"Server not ready yet (attempt {attempt}); retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)

    if not ready:
        logger.error(
            f"Server at {backend_url} did not become ready within {server_wait_s:.0f}s. "
            "Proceeding anyway — requests may fail."
        )

    # --- Model auto-detection ---
    discovered_model = await client.discover_model(timeout=5.0)
    if discovered_model:
        logger.info(f"Server is serving model: {discovered_model}")

    for model in models:
        # Warn and override if the requested model name doesn't match the server's model
        effective_model = model
        if discovered_model and discovered_model != model:
            logger.warning(
                f"Requested model '{model}' does not match server model '{discovered_model}'. "
                f"Using server model '{discovered_model}' for API requests."
            )
            effective_model = discovered_model

        # --- Discover max sequence length for this model ---
        if max_seq_len_override is not None:
            max_seq_len = max_seq_len_override
            logger.info(f"Using user-provided max_seq_len={max_seq_len}")
        else:
            max_seq_len = await _discover_max_seq_len(
                client=client,
                model_path=effective_model,
                backend_url=backend_url,
            )
        logger.info(f"Effective max_seq_len={max_seq_len} for model '{effective_model}'")

        for scenario in scenarios:
            effective_output_tokens = scenario.output_tokens
            if (
                max_output_tokens_override is not None
                and effective_output_tokens > max_output_tokens_override
            ):
                logger.warning(
                    f"Scenario '{scenario.name}': output_tokens clamped "
                    f"{scenario.output_tokens} → {max_output_tokens_override} "
                    "(--max-output-tokens). Metrics reflect this shorter generation."
                )
                effective_output_tokens = max_output_tokens_override

            effective_prompt_tokens = min(
                scenario.prompt_tokens,
                max(10, max_seq_len - effective_output_tokens - 10),
            )
            if effective_prompt_tokens < scenario.prompt_tokens:
                logger.warning(
                    f"Scenario '{scenario.name}': prompt_tokens clamped "
                    f"{scenario.prompt_tokens} → {effective_prompt_tokens} "
                    f"to fit model context window ({max_seq_len} tokens). "
                    "Use --max-seq-len to override."
                )
            logger.info(
                f"Live benchmark: model={effective_model} scenario={scenario.name} "
                f"batch_size={scenario.batch_size} "
                f"prompt_tokens≈{effective_prompt_tokens} "
                f"output_tokens={effective_output_tokens}"
            )
            row = await _run_live_scenario(
                client=client,
                model=effective_model,
                requested_model=model,
                scenario=scenario,
                effective_prompt_tokens=effective_prompt_tokens,
                effective_output_tokens=effective_output_tokens,
            )
            rows.append(row)

    await client.close()
    return rows


async def _run_live_scenario(
    *,
    client: Any,
    model: str,
    requested_model: str,
    scenario: Scenario,
    effective_prompt_tokens: int | None = None,
    effective_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Run a single scenario against a live API server.

    Sends `batch_size` concurrent requests and aggregates the real metrics.

    Args:
        client: GatewayClient instance.
        model: Effective model name used in API requests (may differ from requested_model).
        requested_model: Original model name the user requested (used in result row labeling).
        scenario: Benchmark scenario definition.
        effective_prompt_tokens: Clamped prompt token count (≤ model context window).
            Defaults to scenario.prompt_tokens if not provided.
        effective_output_tokens: Clamped output token count (≤ max_output_tokens cap).
            Defaults to scenario.output_tokens if not provided.

    Returns:
        Result row dict compatible with simulate-mode output.
    """
    from sagellm_benchmark.types import BenchmarkRequest

    # Build a synthetic prompt of approximately the right token length.
    # English text averages ~1.3 tokens/word; we use a safe conservative ratio.
    prompt_tokens = (
        effective_prompt_tokens if effective_prompt_tokens is not None else scenario.prompt_tokens
    )
    output_tokens = (
        effective_output_tokens if effective_output_tokens is not None else scenario.output_tokens
    )
    words_needed = max(10, int(prompt_tokens / 1.3))
    filler_word = "benchmark"
    prompt = " ".join([filler_word] * words_needed)

    requests = [
        BenchmarkRequest(
            prompt=prompt,
            max_tokens=output_tokens,
            request_id=f"live-{scenario.name}-{i}",
            model=model,
            stream=True,
        )
        for i in range(scenario.batch_size)
    ]

    started_at = time.perf_counter()
    results = await client.generate_batch(requests, concurrent=True)
    wall_time_s = max(time.perf_counter() - started_at, 1e-9)

    # Aggregate per-request metrics
    ttft_values: list[float] = []
    tbt_values: list[float] = []
    throughput_values: list[float] = []
    e2e_latencies: list[float] = []
    successful = 0

    for result in results:
        if result.success and result.metrics:
            successful += 1
            m = result.metrics
            ttft_values.append(m.ttft_ms)
            tbt_values.append(m.tbt_ms)
            throughput_values.append(m.throughput_tps)
            # Approximate E2E latency: TTFT + TBT * (output_tokens - 1)
            e2e = m.ttft_ms + m.tbt_ms * max(0, result.output_tokens - 1)
            e2e_latencies.append(e2e)
        else:
            logger.warning(f"Request {result.request_id} failed: {result.error}")

    failed = len(results) - successful
    avg_ttft = mean(ttft_values) if ttft_values else 0.0
    avg_tbt = mean(tbt_values) if tbt_values else 0.0
    avg_throughput = mean(throughput_values) if throughput_values else 0.0
    total_output_tokens = sum(result.output_tokens for result in results if result.success)
    output_throughput_tps = total_output_tokens / wall_time_s if successful else 0.0

    logger.info(
        f"Scenario {scenario.name}: {successful}/{len(results)} ok, "
        f"ttft={avg_ttft:.1f}ms tbt={avg_tbt:.1f}ms throughput={avg_throughput:.1f}tps"
    )

    return {
        "model": requested_model,
        "effective_model": model,
        "precision": "live",
        "scenario": scenario.name,
        "batch_size": scenario.batch_size,
        "ttft_ms": avg_ttft,
        "tbt_ms": avg_tbt,
        "throughput_tps": avg_throughput,
        "output_throughput_tps": output_throughput_tps,
        "latency_p50_ms": _percentile(e2e_latencies, 50),
        "latency_p95_ms": _percentile(e2e_latencies, 95),
        "latency_p99_ms": _percentile(e2e_latencies, 99),
        "memory_mb": 0.0,  # Not available via OpenAI-compatible API
        "mode": "live",
        "successful_requests": successful,
        "failed_requests": failed,
    }


def _stable_seed(*parts: str) -> int:
    joined = "::".join(parts)
    digest = sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = index - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def summarize_e2e_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_rows": len(rows),
        "avg_ttft_ms": mean(row["ttft_ms"] for row in rows) if rows else 0.0,
        "avg_tbt_ms": mean(row["tbt_ms"] for row in rows) if rows else 0.0,
        "avg_throughput_tps": mean(row["throughput_tps"] for row in rows) if rows else 0.0,
        "avg_output_throughput_tps": (
            mean(row.get("output_throughput_tps", row["throughput_tps"]) for row in rows)
            if rows
            else 0.0
        ),
    }
