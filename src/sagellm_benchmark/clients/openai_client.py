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

Uses direct HTTP/SSE requests for benchmark timing.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sagellm_benchmark.clients.base import BenchmarkClient
from sagellm_benchmark.clients.openai_stream import OpenAIStreamBenchmarker

if TYPE_CHECKING:
    from sagellm_benchmark.types import BenchmarkRequest, BenchmarkResult

logger = logging.getLogger(__name__)


def _ensure_hf_endpoint_defaults() -> None:
    """Set benchmark-safe Hugging Face endpoint defaults before hub imports.

    `transformers` / `huggingface_hub` may snapshot endpoint-related environment
    variables during import, so benchmark code must initialize them before any
    lazy model/tokenizer import path.
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


class GatewayClient(BenchmarkClient):
    """Client for OpenAI-protocol HTTP APIs (sagellm-gateway, etc.).

    This client works with any service implementing OpenAI's API protocol.
    Primary use case: Connect to sagellm-gateway for benchmarking.

    Attributes:
        base_url: API base URL (e.g., http://localhost:8000/v1).
        api_key: API key (default: "sagellm-benchmark").
        endpoint_type: Streaming benchmark endpoint family. Only `chat` is supported.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "sagellm-benchmark",
        timeout: float = 60.0,
        endpoint_type: str = "chat",
        http_client: Any | None = None,
        time_fn: Callable[[], float] | None = None,
        zero_content_retry_attempts: int | None = None,
    ) -> None:
        """Initialize Gateway client.

        Args:
            base_url: API base URL.
            api_key: API key.
            timeout: Request timeout (seconds).

        Raises:
            ImportError: If httpx is not installed.
        """
        super().__init__(name="gateway", timeout=timeout)

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx dependency missing. Reinstall benchmark base package with: "
                "pip install -U isagellm-benchmark"
            )

        if endpoint_type != "chat":
            raise ValueError(
                "Streaming benchmark only supports /v1/chat/completions. "
                "/v1/completions with stream=true is outside the benchmark support boundary. "
                f"Received endpoint_type={endpoint_type!r}."
            )

        self.base_url = base_url
        self.api_key = api_key
        self.endpoint_type = endpoint_type
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(timeout=timeout)
        self._tokenizer_cache: dict[str, Any | None] = {}
        if zero_content_retry_attempts is None:
            zero_content_retry_attempts = int(
                os.getenv("SAGELLM_BENCHMARK_STREAM_ZERO_CONTENT_RETRIES", "1")
            )
        if zero_content_retry_attempts < 0:
            logger.warning(
                "Received negative zero_content_retry_attempts=%s; clamping to 0",
                zero_content_retry_attempts,
            )
            zero_content_retry_attempts = 0
        self._stream_benchmarker = OpenAIStreamBenchmarker(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            endpoint_type=endpoint_type,
            http_client=self._http_client,
            time_fn=time_fn,
            token_counter=self._count_text_tokens,
            zero_content_retry_attempts=zero_content_retry_attempts,
        )

        logger.info(
            "Gateway client initialized: base_url=%s endpoint_type=%s", base_url, endpoint_type
        )

    async def generate(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Execute request via OpenAI API.

        Args:
            request: Benchmark request.

        Returns:
            Benchmark result with metrics.
        """
        return await self._stream_benchmarker.benchmark(request)

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

        _ensure_hf_endpoint_defaults()
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

        import httpx

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
        if self._owns_http_client:
            await self._http_client.aclose()
        logger.info("Gateway client closed")
