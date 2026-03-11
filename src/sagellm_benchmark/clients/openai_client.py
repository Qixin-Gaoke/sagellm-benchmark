"""Gateway client for OpenAI-protocol HTTP APIs.

This client connects to any service using OpenAI's API protocol:
- sagellm-gateway (primary use case - local sageLLM deployment)
- sagellm-core engine_server (direct engine HTTP server)
- OpenAI API (cloud - for comparison benchmarks)
- vLLM OpenAI server
- LMDeploy OpenAI server
- Other OpenAI-compatible endpoints

Note: This is NOT OpenAI-specific. It's a generic client for
OpenAI-protocol APIs. For sageLLM benchmarks, use this to
connect to sagellm-gateway.

Uses the official openai Python SDK.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sagellm_benchmark.clients.base import BenchmarkClient

if TYPE_CHECKING:
    from sagellm_benchmark.types import BenchmarkRequest, BenchmarkResult

logger = logging.getLogger(__name__)


class GatewayClient(BenchmarkClient):
    """Client for OpenAI-protocol HTTP APIs (sagellm-gateway, etc.).

    This client works with any service implementing OpenAI's API protocol.
    Primary use case: Connect to sagellm-gateway for benchmarking.

    Attributes:
        base_url: API base URL (e.g., http://localhost:8000/v1).
        api_key: API key (default: "sagellm-benchmark").
        client: OpenAI async client instance.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "sagellm-benchmark",
        timeout: float = 60.0,
    ) -> None:
        """Initialize Gateway client.

        Args:
            base_url: API base URL.
            api_key: API key.
            timeout: Request timeout (seconds).

        Raises:
            ImportError: If openai package not installed.
        """
        super().__init__(name="gateway", timeout=timeout)

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai dependency missing. Reinstall benchmark base package with: "
                "pip install -U isagellm-benchmark"
            )

        self.base_url = base_url
        self.api_key = api_key
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._tokenizer_cache: dict[str, Any | None] = {}

        logger.info(f"OpenAI client initialized: base_url={base_url}")

    async def generate(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Execute request via OpenAI API.

        Args:
            request: Benchmark request.

        Returns:
            Benchmark result with metrics.
        """
        from sagellm_benchmark.types import BenchmarkResult

        try:
            from sagellm_protocol import Metrics
        except ImportError:
            logger.error("sagellm_protocol not installed")
            return BenchmarkResult(
                request_id=request.request_id,
                success=False,
                error="sagellm_protocol not installed",
                metrics=None,
            )

        start_time = time.perf_counter()
        first_token_time = None
        streamed_chunks = 0

        try:
            # Build API request
            api_kwargs: dict[str, Any] = {
                "model": request.model,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens,
                "stream": True,  # Always stream for metrics collection
            }

            if request.temperature is not None:
                api_kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                api_kwargs["top_p"] = request.top_p

            # Execute streaming request
            output_text = ""
            async for chunk in await self.client.chat.completions.create(**api_kwargs):
                current_time = time.perf_counter()

                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    output_text += content
                    streamed_chunks += 1

                    if first_token_time is None:
                        first_token_time = current_time
                        logger.debug(
                            f"TTFT for {request.request_id}: "
                            f"{(first_token_time - start_time) * 1000:.2f}ms"
                        )

            end_time = time.perf_counter()

            # Calculate metrics
            total_time_s = end_time - start_time
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0.0
            output_tokens = self._count_text_tokens(output_text, request.model) or streamed_chunks
            prompt_tokens = self._count_text_tokens(request.prompt, request.model)
            if prompt_tokens <= 0:
                prompt_tokens = len(request.prompt.split())

            # Calculate TBT using real output token count when available.
            if first_token_time is not None and output_tokens > 1:
                tbt_ms = ((end_time - first_token_time) * 1000) / (output_tokens - 1)
            else:
                tbt_ms = 0.0

            # TPOT (time per output token)
            tpot_ms = (total_time_s * 1000 / output_tokens) if output_tokens > 0 else 0.0

            # Throughput
            throughput_tps = output_tokens / total_time_s if total_time_s > 0 else 0.0

            # Create metrics (OpenAI API doesn't provide all metrics)
            metrics = Metrics(
                ttft_ms=ttft_ms,
                tbt_ms=tbt_ms,
                tpot_ms=tpot_ms,
                throughput_tps=throughput_tps,
                peak_mem_mb=0,
                error_rate=0.0,
                # Unavailable from OpenAI API
                kv_used_tokens=0,
                kv_used_bytes=0,
                prefix_hit_rate=0.0,
                evict_count=0,
                evict_ms=0.0,
                spec_accept_rate=0.0,
            )

            return BenchmarkResult(
                request_id=request.request_id,
                success=True,
                error=None,
                metrics=metrics,
                output_text=output_text,
                output_tokens=output_tokens,
                prompt_tokens=prompt_tokens,
            )

        except Exception as e:
            logger.error(f"OpenAI request {request.request_id} failed: {e}", exc_info=True)
            return BenchmarkResult(
                request_id=request.request_id,
                success=False,
                error=str(e),
                metrics=None,
            )

    def _count_text_tokens(self, text: str, model_id: str) -> int:
        """Count real tokenizer tokens for benchmark accounting when tokenizer is available."""
        if not text:
            return 0
        tokenizer = self._get_tokenizer(model_id)
        if tokenizer is None:
            return 0
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(text)
        return len(token_ids)

    def _get_tokenizer(self, model_id: str) -> Any | None:
        """Lazily load and cache a tokenizer for accurate token accounting.

        Uses local/cache-only lookup to avoid hidden network dependency in benchmark runs.
        """
        if not model_id:
            return None
        if model_id in self._tokenizer_cache:
            return self._tokenizer_cache[model_id]

        try:
            from transformers import AutoTokenizer
        except ImportError:
            logger.warning(
                "transformers not available; falling back to streamed chunk counts for model=%s",
                model_id,
            )
            self._tokenizer_cache[model_id] = None
            return None

        tokenizer_source, local_only = self._resolve_tokenizer_source(model_id)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=True,
                local_files_only=local_only,
            )
        except Exception as exc:
            logger.warning(
                "Tokenizer unavailable for model=%s source=%s local_only=%s; "
                "falling back to streamed chunk counts: %s",
                model_id,
                tokenizer_source,
                local_only,
                exc,
            )
            tokenizer = None

        self._tokenizer_cache[model_id] = tokenizer
        return tokenizer

    @staticmethod
    def _resolve_tokenizer_source(model_id: str) -> tuple[str, bool]:
        """Resolve the tokenizer source for benchmark token accounting.

        Priority:
        1. Explicit local model directory env vars.
        2. Direct filesystem path in ``model_id``.
        3. Conventional local cache under ``~/.cache/hf-local-models``.
        4. Remote model id via HuggingFace mirror / configured endpoint.
        """
        explicit_local_dir = (
            os.getenv("SAGELLM_BENCHMARK_LOCAL_MODEL_DIR")
            or os.getenv("VLLM_LOCAL_MODEL_DIR")
            or os.getenv("HF_LOCAL_MODEL_DIR")
        )
        if explicit_local_dir and os.path.exists(explicit_local_dir):
            return explicit_local_dir, True

        if os.path.exists(model_id):
            return model_id, True

        normalized = model_id.strip().strip("/")
        model_leaf = normalized.split("/")[-1] if normalized else model_id
        local_cache_candidates = [
            Path.home() / ".cache" / "hf-local-models" / model_leaf,
            Path.home() / ".cache" / "hf-local-models" / normalized,
        ]
        for candidate in local_cache_candidates:
            if candidate.exists():
                return str(candidate), True

        return model_id, False

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if API is reachable.

        Tries endpoints in priority order:
        1. GET /health  (sagellm-core engine_server)
        2. GET /v1/models  (standard OpenAI-compatible, e.g. vLLM)

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            True if the server is reachable and ready.
        """
        base = self.base_url.rstrip("/")
        # Strip /v1 suffix to get server root
        root = base[:-3] if base.endswith("/v1") else base

        try:
            import httpx
        except ImportError:
            # Fall back to OpenAI SDK /v1/models probe
            return await self._health_check_openai_sdk()

        # 1. Try /health first (sagellm engine_server style)
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                r = await http.get(f"{root}/health")
            if r.status_code < 500:
                logger.info(f"Health check OK via /health (HTTP {r.status_code})")
                return True
        except Exception as e:
            logger.debug(f"/health probe failed: {e}")

        # 2. Try /v1/models (standard OpenAI-compatible)
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                r = await http.get(f"{base}/models")
            if r.status_code < 500:
                logger.info(f"Health check OK via /v1/models (HTTP {r.status_code})")
                return True
        except Exception as e:
            logger.debug(f"/v1/models probe failed: {e}")

        logger.error("All health check probes failed — server may not be ready")
        return False

    async def _health_check_openai_sdk(self) -> bool:
        """Fallback health check using the OpenAI SDK models.list()."""
        try:
            models = await asyncio.wait_for(self.client.models.list(), timeout=10.0)
            logger.info(f"Health check OK ({len(list(models.data))} models available)")
            return True
        except Exception as e:
            logger.error(f"SDK health check failed: {e}")
            return False

    async def discover_model(self, timeout: float = 5.0) -> str | None:
        """Discover the model name loaded by the server.

        Queries /info (sagellm engine_server) or /v1/models.
        Returns the first model name found, or None.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            Model name string, or None if undetectable.
        """
        base = self.base_url.rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base

        try:
            import httpx
        except ImportError:
            return None

        # Try /info first (sagellm engine_server exposes model_path here)
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                r = await http.get(f"{root}/info")
            if r.status_code == 200:
                data = r.json()
                model = data.get("model_path") or data.get("model") or data.get("model_name")
                if model:
                    logger.info(f"Discovered model from /info: {model}")
                    return str(model)
        except Exception as e:
            logger.debug(f"/info probe failed: {e}")

        # Try /v1/models
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                r = await http.get(f"{base}/models")
            if r.status_code == 200:
                data = r.json()
                models = data.get("data", [])
                if models:
                    model = models[0].get("id")
                    if model:
                        logger.info(f"Discovered model from /v1/models: {model}")
                        return str(model)
        except Exception as e:
            logger.debug(f"/v1/models model discovery failed: {e}")

        return None

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.close()
        logger.info("OpenAI client closed")
