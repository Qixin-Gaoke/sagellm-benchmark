"""Streaming benchmark helpers for chat-completions benchmark endpoints.

This module measures request-level latency directly from SSE responses without
depending on the OpenAI SDK streaming abstractions. It is intentionally scoped
to benchmark collection rather than CLI orchestration.
"""

from __future__ import annotations

import codecs
import json
import logging
import time
from statistics import mean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    import httpx

    from sagellm_benchmark.types import BenchmarkRequest, BenchmarkResult

logger = logging.getLogger(__name__)


class OpenAIStreamBenchmarker:
    """Collect streaming latency metrics from chat-completions SSE endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        endpoint_type: str = "chat",
        http_client: httpx.AsyncClient | Any,
        time_fn: Callable[[], float] | None = None,
        token_counter: Callable[[str, str], int] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.endpoint_type = endpoint_type
        self._http_client = http_client
        self._time_fn = time_fn or time.perf_counter
        self._token_counter = token_counter
        self._validate_endpoint_type()

    @staticmethod
    def _unsupported_endpoint_message(endpoint_type: str) -> str:
        return (
            "Streaming benchmark only supports /v1/chat/completions. "
            "/v1/completions with stream=true is outside the benchmark support boundary. "
            f"Received endpoint_type={endpoint_type!r}."
        )

    def _validate_endpoint_type(self) -> None:
        if self.endpoint_type != "chat":
            raise ValueError(self._unsupported_endpoint_message(self.endpoint_type))

    async def benchmark(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Execute one streaming benchmark request and map it into BenchmarkResult."""
        from sagellm_protocol import Metrics, Timestamps

        from sagellm_benchmark.types import BenchmarkResult

        started_at = self._time_fn()
        first_token_at: float | None = None
        last_token_at: float | None = None
        output_parts: list[str] = []
        itl_list: list[float] = []
        content_events = 0
        usage_prompt_tokens = 0
        usage_output_tokens = 0

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

        try:
            async with self._http_client.stream(
                "POST",
                self._endpoint_url(),
                headers=headers,
                json=self._build_payload(request),
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    return BenchmarkResult(
                        request_id=request.request_id,
                        success=False,
                        error=self._format_http_error(response.status_code, body),
                        metrics=None,
                    )

                async for event_data in self._iter_sse_data(response.aiter_bytes()):
                    if not event_data or event_data == "[DONE]":
                        continue

                    payload = json.loads(event_data)
                    usage_prompt_tokens, usage_output_tokens = self._merge_usage_counts(
                        payload=payload,
                        prompt_tokens=usage_prompt_tokens,
                        output_tokens=usage_output_tokens,
                    )

                    content = self._extract_content(payload)
                    if not content:
                        continue

                    now = self._time_fn()
                    output_parts.append(content)
                    content_events += 1

                    if first_token_at is None:
                        first_token_at = now
                    elif last_token_at is not None:
                        itl_list.append((now - last_token_at) * 1000.0)

                    last_token_at = now

        except Exception as exc:
            logger.error("Streaming request %s failed: %s", request.request_id, exc, exc_info=True)
            return BenchmarkResult(
                request_id=request.request_id,
                success=False,
                error=str(exc),
                metrics=None,
            )

        completed_at = self._time_fn()
        output_text = "".join(output_parts)
        output_tokens = self._count_text_tokens(output_text, request.model)
        if output_tokens <= 0:
            output_tokens = usage_output_tokens
        if output_tokens <= 0:
            output_tokens = content_events

        prompt_tokens = self._count_text_tokens(request.prompt, request.model)
        if prompt_tokens <= 0:
            prompt_tokens = usage_prompt_tokens
        if prompt_tokens <= 0:
            prompt_tokens = len(request.prompt.split())

        ttft_ms = ((first_token_at - started_at) * 1000.0) if first_token_at is not None else 0.0
        e2e_latency_ms = (completed_at - started_at) * 1000.0
        tbt_ms = mean(itl_list) if itl_list else 0.0

        if output_tokens > 1 and first_token_at is not None:
            tpot_ms = ((completed_at - first_token_at) * 1000.0) / (output_tokens - 1)
        elif output_tokens == 1:
            tpot_ms = ttft_ms
        else:
            tpot_ms = 0.0

        throughput_tps = (
            output_tokens / (completed_at - started_at) if completed_at > started_at else 0.0
        )

        metrics = Metrics(
            ttft_ms=ttft_ms,
            tbt_ms=tbt_ms,
            tpot_ms=tpot_ms,
            throughput_tps=throughput_tps,
            peak_mem_mb=0,
            error_rate=0.0,
            kv_used_tokens=0,
            kv_used_bytes=0,
            prefix_hit_rate=0.0,
            evict_count=0,
            evict_ms=0.0,
            spec_accept_rate=0.0,
            itl_list=itl_list,
            timestamps=Timestamps(
                queued_at=started_at,
                scheduled_at=started_at,
                executed_at=first_token_at or started_at,
                completed_at=completed_at,
            ),
        )

        return BenchmarkResult(
            request_id=request.request_id,
            success=True,
            error=None,
            metrics=metrics,
            output_text=output_text,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            itl_list=itl_list,
            e2e_latency_ms=e2e_latency_ms,
        )

    def _endpoint_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _build_payload(self, request: BenchmarkRequest) -> dict[str, Any]:
        return {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "stream": True,
            # Compare mainline only depends on chat streaming and local metric accounting.
            "messages": [{"role": "user", "content": request.prompt}],
        }

    async def _iter_sse_data(self, byte_iter: AsyncIterator[bytes]) -> AsyncIterator[str]:
        decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""

        async for chunk in byte_iter:
            if not chunk:
                continue
            buffer += decoder.decode(chunk)
            buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")

            while True:
                boundary = buffer.find("\n\n")
                if boundary < 0:
                    break
                raw_event = buffer[:boundary]
                buffer = buffer[boundary + 2 :]
                event_data = self._decode_sse_event(raw_event)
                if event_data is not None:
                    yield event_data

        buffer += decoder.decode(b"", final=True)
        buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")

        while True:
            boundary = buffer.find("\n\n")
            if boundary < 0:
                break
            raw_event = buffer[:boundary]
            buffer = buffer[boundary + 2 :]
            event_data = self._decode_sse_event(raw_event)
            if event_data is not None:
                yield event_data

    @staticmethod
    def _decode_sse_event(raw_event: str) -> str | None:
        data_lines: list[str] = []
        for line in raw_event.split("\n"):
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:]
            if data.startswith(" "):
                data = data[1:]
            data_lines.append(data)

        if not data_lines:
            return None

        return "\n".join(data_lines)

    def _count_text_tokens(self, text: str, model_id: str) -> int:
        if not text or self._token_counter is None:
            return 0
        return self._token_counter(text, model_id)

    def _extract_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        parts: list[str] = []

        for choice in choices:
            delta = choice.get("delta") or {}
            content = delta.get("content")

            if isinstance(content, str) and content:
                parts.append(content)

        return "".join(parts)

    @staticmethod
    def _merge_usage_counts(
        *,
        payload: dict[str, Any],
        prompt_tokens: int,
        output_tokens: int,
    ) -> tuple[int, int]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return prompt_tokens, output_tokens

        prompt_value = usage.get("prompt_tokens")
        completion_value = usage.get("completion_tokens")

        if isinstance(prompt_value, int) and prompt_value > 0:
            prompt_tokens = prompt_value
        if isinstance(completion_value, int) and completion_value > 0:
            output_tokens = completion_value

        return prompt_tokens, output_tokens

    @staticmethod
    def _format_http_error(status_code: int, body: bytes) -> str:
        body_text = body.decode("utf-8", errors="replace").strip()
        message = body_text
        try:
            payload = json.loads(body_text) if body_text else {}
        except json.JSONDecodeError:
            payload = {}

        error_payload = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error_payload, dict) and error_payload.get("message"):
            message = str(error_payload["message"])
        elif isinstance(payload, dict) and payload.get("message"):
            message = str(payload["message"])

        message = message or "empty response body"
        return f"HTTP {status_code}: {message}"
