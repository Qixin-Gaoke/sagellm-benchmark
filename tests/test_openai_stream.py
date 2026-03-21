from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from sagellm_benchmark.clients.openai_client import GatewayClient
from sagellm_benchmark.clients.openai_stream import OpenAIStreamBenchmarker
from sagellm_benchmark.metrics import MetricsAggregator
from sagellm_benchmark.types import BenchmarkRequest


class _FakeResponse:
    def __init__(
        self, *, status_code: int, chunks: list[bytes | Exception], body: bytes = b""
    ) -> None:
        self.status_code = status_code
        self._chunks = chunks
        self._body = body

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def aread(self) -> bytes:
        return self._body


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


class _FakeHTTPClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def stream(self, method: str, url: str, **kwargs) -> _FakeStreamContext:
        self.calls.append({"method": method, "url": url, **kwargs})
        return _FakeStreamContext(self.response)

    async def aclose(self) -> None:
        return None


class _SequentialHTTPClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []
        self._index = 0

    def stream(self, method: str, url: str, **kwargs) -> _FakeStreamContext:
        if self._index >= len(self._responses):
            raise RuntimeError("no more fake responses configured")
        response = self._responses[self._index]
        self._index += 1
        self.calls.append({"method": method, "url": url, **kwargs})
        return _FakeStreamContext(response)

    async def aclose(self) -> None:
        return None


def _request() -> BenchmarkRequest:
    return BenchmarkRequest(
        prompt="Hello benchmark",
        max_tokens=16,
        request_id="req-001",
        model="demo-model",
    )


@pytest.mark.asyncio
async def test_chat_stream_handles_fragmented_sse_and_usage_chunk() -> None:
    stream_bytes = (
        b'data: {"event":"start","trace_id":"trace-stream-001","choices":[{"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"event":"delta","trace_id":"trace-stream-001","choices":[{"delta":{"content":"Hel"}}]}\n\n'
        b'data: {"event":"delta","trace_id":"trace-stream-001","choices":[{"delta":{"content":"lo"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}\n\n'
        b"data: [DONE]\n\n"
    )
    response = _FakeResponse(
        status_code=200,
        chunks=[stream_bytes[:17], stream_bytes[17:49], stream_bytes[49:82], stream_bytes[82:]],
    )
    client = _FakeHTTPClient(response)
    clock = iter([10.0, 10.05, 10.08, 10.12])

    bench = OpenAIStreamBenchmarker(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token",
        http_client=client,
        time_fn=lambda: next(clock),
        token_counter=lambda text, model: len(text.replace(" ", "")),
    )

    result = await bench.benchmark(_request())

    assert result.success is True
    assert result.output_text == "Hello"
    assert result.prompt_tokens == 14
    assert result.output_tokens == 5
    assert result.itl_list == pytest.approx([30.0])
    assert result.e2e_latency_ms == pytest.approx(120.0)
    assert result.metrics is not None
    assert result.metrics.ttft_ms == pytest.approx(50.0)
    assert result.metrics.tbt_ms == pytest.approx(30.0)
    assert result.metrics.tpot_ms == pytest.approx(17.5)
    assert result.metrics.itl_list == pytest.approx([30.0])
    assert result.trace_id == "trace-stream-001"
    assert result.protocol_surface == "chat_mainline"

    aggregated = MetricsAggregator.aggregate([result])
    assert aggregated.avg_ttft_ms == pytest.approx(50.0)
    assert aggregated.avg_itl_ms == pytest.approx(30.0)
    assert aggregated.avg_e2el_ms == pytest.approx(120.0)
    assert aggregated.total_output_tokens == 5


@pytest.mark.asyncio
async def test_chat_stream_ignores_empty_chunks() -> None:
    response = _FakeResponse(
        status_code=200,
        chunks=[
            b"data: \n\n",
            b'data: {"choices":[{"delta":{"content":""}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    client = _FakeHTTPClient(response)
    clock = iter([20.0, 20.04, 20.10])

    bench = OpenAIStreamBenchmarker(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token",
        http_client=client,
        time_fn=lambda: next(clock),
        token_counter=lambda text, model: len(text),
    )

    result = await bench.benchmark(_request())

    assert result.success is True
    assert result.output_text == "Hi"
    assert result.output_tokens == 2
    assert result.itl_list == []
    assert result.metrics is not None
    assert result.metrics.ttft_ms == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_chat_stream_fails_fast_when_no_content_tokens() -> None:
    response = _FakeResponse(
        status_code=200,
        chunks=[
            b'data: {"choices":[{"delta":{"role":"assistant","content":null},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"role":null,"content":null},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    client = _FakeHTTPClient(response)

    bench = OpenAIStreamBenchmarker(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token",
        http_client=client,
        token_counter=lambda text, model: len(text),
    )

    result = await bench.benchmark(_request())

    assert result.success is False
    assert result.metrics is None
    assert result.error is not None
    assert "stream completed without content delta tokens" in result.error


@pytest.mark.asyncio
async def test_chat_stream_retries_once_after_empty_content_stream() -> None:
    empty_response = _FakeResponse(
        status_code=200,
        chunks=[
            b'data: {"choices":[{"delta":{"role":"assistant","content":null},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"role":null,"content":null},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    content_response = _FakeResponse(
        status_code=200,
        chunks=[
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    client = _SequentialHTTPClient([empty_response, content_response])
    clock = iter([40.0, 40.05, 40.08, 40.10, 40.12])

    bench = OpenAIStreamBenchmarker(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token",
        http_client=client,
        time_fn=lambda: next(clock),
        token_counter=lambda text, model: len(text),
        zero_content_retry_attempts=1,
    )

    result = await bench.benchmark(_request())

    assert result.success is True
    assert result.output_text == "Hi"
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_chat_stream_uses_minimal_payload_without_usage_dependency() -> None:
    response = _FakeResponse(
        status_code=200,
        chunks=[
            b'data: {"choices":[{"delta":{"content":"He"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    client = _FakeHTTPClient(response)
    clock = iter([50.0, 50.03, 50.07, 50.10])
    request = BenchmarkRequest(
        prompt="Hello benchmark",
        max_tokens=16,
        request_id="req-minimal",
        model="demo-model",
        temperature=0.7,
        top_p=0.9,
    )

    bench = OpenAIStreamBenchmarker(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token",
        http_client=client,
        time_fn=lambda: next(clock),
        token_counter=lambda text, model: len(text.replace(" ", "")),
    )

    result = await bench.benchmark(request)

    assert result.success is True
    assert result.prompt_tokens == 14
    assert result.output_tokens == 5
    assert result.metrics is not None
    assert result.metrics.ttft_ms == pytest.approx(30.0)
    assert result.metrics.tbt_ms == pytest.approx(40.0)
    assert client.calls[0]["json"] == {
        "model": "demo-model",
        "max_tokens": 16,
        "stream": True,
        "messages": [{"role": "user", "content": "Hello benchmark"}],
    }


@pytest.mark.asyncio
async def test_chat_stream_fails_fast_on_explicit_error_event() -> None:
    response = _FakeResponse(
        status_code=200,
        chunks=[
            b'data: {"event":"start","trace_id":"trace-stream-error","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"event":"error","trace_id":"trace-stream-error","error":{"message":"gateway upstream failed","code":"unavailable"},"choices":[{"delta":{},"finish_reason":"error"}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    client = _FakeHTTPClient(response)

    bench = OpenAIStreamBenchmarker(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token",
        http_client=client,
        token_counter=lambda text, model: len(text),
    )

    result = await bench.benchmark(_request())

    assert result.success is False
    assert result.error == "gateway upstream failed"
    assert result.error_code == "unavailable"
    assert result.trace_id == "trace-stream-error"


@pytest.mark.asyncio
async def test_chat_stream_keeps_chat_only_minimal_payload_for_strict_endpoints() -> None:
    response = _FakeResponse(
        status_code=200,
        chunks=[
            b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    class _StrictHTTPClient(_FakeHTTPClient):
        def stream(self, method: str, url: str, **kwargs) -> _FakeStreamContext:
            assert url.endswith("/chat/completions"), (
                "chat streaming benchmark boundary reopened: request path drifted away from "
                "/chat/completions"
            )
            forbidden_fields = {
                key
                for key in kwargs["json"]
                if key in {"temperature", "top_p", "stream_options", "n", "best_of"}
            }
            assert not forbidden_fields, (
                "strict endpoint boundary reopened: benchmark started sending forbidden default "
                f"payload fields {sorted(forbidden_fields)}"
            )
            assert kwargs["json"] == {
                "model": "strict-model",
                "max_tokens": 4,
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            }, (
                "strict endpoint boundary reopened: benchmark payload is no longer minimal chat streaming"
            )
            assert set(kwargs["headers"].keys()) == {"Authorization", "Accept", "Content-Type"}
            return super().stream(method, url, **kwargs)

    client = _StrictHTTPClient(response)
    bench = OpenAIStreamBenchmarker(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token",
        http_client=client,
        token_counter=lambda text, model: len(text),
    )

    result = await bench.benchmark(
        BenchmarkRequest(
            prompt="ping",
            max_tokens=4,
            request_id="req-strict",
            model="strict-model",
            temperature=1.5,
            top_p=0.2,
        )
    )

    assert result.success is True
    assert result.output_text == "OK"


@pytest.mark.asyncio
async def test_stream_failure_on_midstream_interrupt() -> None:
    response = _FakeResponse(
        status_code=200,
        chunks=[
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
            RuntimeError("connection dropped"),
        ],
    )
    client = _FakeHTTPClient(response)
    clock = iter([30.0, 30.03])

    bench = OpenAIStreamBenchmarker(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token",
        http_client=client,
        time_fn=lambda: next(clock),
        token_counter=lambda text, model: len(text),
    )

    result = await bench.benchmark(_request())

    assert result.success is False
    assert result.metrics is None
    assert result.error is not None
    assert "connection dropped" in result.error


@pytest.mark.asyncio
async def test_stream_failure_on_non_200_response() -> None:
    response = _FakeResponse(
        status_code=503,
        chunks=[],
        body=b'{"error":{"message":"server overloaded"}}',
    )
    client = _FakeHTTPClient(response)

    bench = OpenAIStreamBenchmarker(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token",
        http_client=client,
    )

    result = await bench.benchmark(_request())

    assert result.success is False
    assert result.metrics is None
    assert result.error == "HTTP 503: server overloaded"


def test_openai_stream_rejects_completions_streaming_endpoint_type() -> None:
    response = _FakeResponse(status_code=200, chunks=[])
    client = _FakeHTTPClient(response)

    with pytest.raises(
        ValueError,
        match=r"Streaming benchmark only supports /v1/chat/completions.*outside the benchmark support boundary",
    ):
        OpenAIStreamBenchmarker(
            base_url="http://127.0.0.1:8000/v1",
            api_key="token",
            endpoint_type="completions",
            http_client=client,
        )


def test_gateway_client_rejects_completions_streaming_endpoint_type() -> None:
    with pytest.raises(
        ValueError,
        match=r"Streaming benchmark only supports /v1/chat/completions.*outside the benchmark support boundary",
    ):
        GatewayClient(
            base_url="http://127.0.0.1:8000/v1",
            api_key="token",
            endpoint_type="completions",
        )
