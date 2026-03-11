"""Tests for benchmark clients."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_helpers import StubClient

from sagellm_benchmark.clients import BenchmarkClient
from sagellm_benchmark.clients.openai_client import GatewayClient
from sagellm_benchmark.types import BenchmarkRequest


@pytest.fixture
def sample_request() -> BenchmarkRequest:
    """Create a sample benchmark request."""
    return BenchmarkRequest(
        prompt="What is the capital of France?",
        max_tokens=50,
        request_id="test-001",
        model="cpu-model",
        temperature=0.7,
        top_p=0.9,
    )


@pytest.fixture
def batch_requests() -> list[BenchmarkRequest]:
    """Create a batch of benchmark requests."""
    return [
        BenchmarkRequest(
            prompt=f"Question {i}",
            max_tokens=20,
            request_id=f"test-{i:03d}",
            model="cpu-model",
        )
        for i in range(5)
    ]


class TestStubClient:  # Renamed from TestMockClient
    """Tests for StubClient."""

    @pytest.mark.asyncio
    async def test_single_request(self, sample_request: BenchmarkRequest) -> None:
        """Test single request execution."""
        client = StubClient(ttft_ms=10.0, tbt_ms=5.0, throughput_tps=100.0)

        result = await client.generate(sample_request)

        assert result.success
        assert result.error is None
        assert result.request_id == sample_request.request_id
        assert result.metrics is not None
        assert result.metrics.ttft_ms == 10.0
        assert result.metrics.tbt_ms == 5.0
        assert result.output_tokens == sample_request.max_tokens

    @pytest.mark.asyncio
    async def test_sequential_batch(self, batch_requests: list[BenchmarkRequest]) -> None:
        """Test sequential batch execution."""
        client = StubClient(ttft_ms=5.0, tbt_ms=2.0)

        results = await client.generate_batch(batch_requests, concurrent=False)

        assert len(results) == len(batch_requests)
        for i, result in enumerate(results):
            assert result.request_id == batch_requests[i].request_id
            assert result.success

    @pytest.mark.asyncio
    async def test_concurrent_batch(self, batch_requests: list[BenchmarkRequest]) -> None:
        """Test concurrent batch execution."""
        client = StubClient(ttft_ms=5.0, tbt_ms=2.0)

        results = await client.generate_batch(batch_requests, concurrent=True)

        assert len(results) == len(batch_requests)
        # Verify order preservation
        for i, result in enumerate(results):
            assert result.request_id == batch_requests[i].request_id
            assert result.success

    @pytest.mark.asyncio
    async def test_error_simulation(self) -> None:
        """Test error simulation."""
        client = StubClient(error_rate=1.0)  # 100% failure rate

        request = BenchmarkRequest(
            prompt="Test",
            max_tokens=10,
            request_id="error-test",
        )

        result = await client.generate(request)

        assert not result.success
        assert result.error is not None
        assert "Simulated failure" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        """Test timeout handling."""
        # Create a very slow client with short timeout
        client = StubClient(ttft_ms=1000.0, tbt_ms=1000.0, timeout=0.1)

        request = BenchmarkRequest(
            prompt="Test",
            max_tokens=100,
            request_id="timeout-test",
        )

        result = await client.generate(request)

        assert not result.success
        assert result.error is not None
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health check."""
        client = StubClient()

        is_healthy = await client.health_check()

        assert is_healthy


class TestBenchmarkClientInterface:
    """Tests for BenchmarkClient abstract interface."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            BenchmarkClient()  # type: ignore

    @pytest.mark.asyncio
    async def test_custom_client_implementation(self) -> None:
        """Test custom client implementation."""

        class CustomClient(BenchmarkClient):
            async def generate(self, request: BenchmarkRequest) -> BenchmarkRequest:  # type: ignore
                from sagellm_protocol import Metrics

                from sagellm_benchmark.types import BenchmarkResult

                return BenchmarkResult(
                    request_id=request.request_id,
                    success=True,
                    error=None,
                    metrics=Metrics(
                        ttft_ms=1.0,
                        tbt_ms=1.0,
                        tpot_ms=1.0,
                        throughput_tps=1.0,
                        peak_mem_mb=0,
                        error_rate=0.0,
                        kv_used_tokens=0,
                        kv_used_bytes=0,
                        prefix_hit_rate=0.0,
                        evict_count=0,
                        evict_ms=0.0,
                        spec_accept_rate=0.0,
                    ),
                )

        client = CustomClient(name="custom")

        request = BenchmarkRequest(
            prompt="Test",
            max_tokens=10,
            request_id="custom-001",
        )

        result = await client.generate(request)

        assert result.success
        assert result.request_id == "custom-001"


@pytest.mark.asyncio
async def test_batch_order_preservation() -> None:
    """Test that batch results preserve input order."""
    client = StubClient()

    requests = [
        BenchmarkRequest(
            prompt=f"Request {i}",
            max_tokens=10,
            request_id=f"order-{i:03d}",
        )
        for i in range(10)
    ]

    # Test both modes
    for concurrent in [True, False]:
        results = await client.generate_batch(requests, concurrent=concurrent)

        assert len(results) == len(requests)
        for i, result in enumerate(results):
            assert result.request_id == f"order-{i:03d}"


@pytest.mark.asyncio
async def test_batch_partial_failure() -> None:
    """Test batch execution with partial failures."""
    # 50% failure rate
    client = StubClient(error_rate=0.5)

    requests = [
        BenchmarkRequest(
            prompt=f"Request {i}",
            max_tokens=10,
            request_id=f"partial-{i:03d}",
        )
        for i in range(20)
    ]

    results = await client.generate_batch(requests, concurrent=True)

    assert len(results) == len(requests)

    successes = sum(1 for r in results if r.success)
    failures = sum(1 for r in results if not r.success)

    # With 50% error rate, expect roughly half to fail
    assert failures > 0
    assert successes > 0
    assert successes + failures == len(requests)


# ==================== ITL/E2EL Tests ====================


@pytest.mark.asyncio
async def test_simulated_itl_generation(sample_request: BenchmarkRequest) -> None:
    """Test that StubClient generates ITL list and E2E latency."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0, throughput_tps=100.0)

    result = await client.generate(sample_request)

    assert result.success
    assert result.error is None

    # 验证 itl_list 非空且长度等于 output_tokens
    assert result.itl_list is not None
    assert len(result.itl_list) == sample_request.max_tokens

    # 验证 ITL 值在合理范围
    # First token should be TTFT (10.0ms), rest should be TBT (5.0ms)
    assert result.itl_list[0] == 10.0  # TTFT
    for itl in result.itl_list[1:]:  # Remaining tokens
        assert itl == 5.0  # TBT

    # 验证 e2e_latency_ms 大于 0
    assert result.e2e_latency_ms > 0

    # E2E 应该至少包含 TTFT + 总生成时间
    min_expected_e2e = 10.0 + 5.0 * sample_request.max_tokens * 0.8
    assert result.e2e_latency_ms >= min_expected_e2e * 0.9  # 允许一些误差


@pytest.mark.asyncio
async def test_simulated_full_itl_mode() -> None:
    """Test simulated client with simulate_full_itl=True for realistic simulation."""
    client = StubClient(ttft_ms=5.0, tbt_ms=2.0, simulate_full_itl=True)

    request = BenchmarkRequest(
        prompt="Test",
        max_tokens=5,  # 少量 token 以加速测试
        request_id="full-itl-test",
    )

    result = await client.generate(request)

    assert result.success
    assert len(result.itl_list) == 5

    # 在 full_itl 模式下，ITL 应该是实际测量值
    for itl in result.itl_list:
        assert itl > 0
        # 只验证下界（系统调度可能导致上界波动）
        assert itl >= 1.5  # 至少应该接近 tbt_ms * 0.8

    # 验证平均 ITL 接近 tbt_ms（允许较大波动）
    avg_itl = sum(result.itl_list) / len(result.itl_list)
    assert 1.5 <= avg_itl <= 5.0  # 平均值应该接近 2.0ms，允许系统波动

    # E2E latency 应该大于 TTFT + 所有 ITL 的和
    assert result.e2e_latency_ms > 5.0


@pytest.mark.asyncio
async def test_simulated_itl_on_failure() -> None:
    """Test that failed requests don't have ITL/E2E data."""
    client = StubClient(error_rate=1.0)  # 100% failure

    request = BenchmarkRequest(
        prompt="Test",
        max_tokens=10,
        request_id="fail-test",
    )

    result = await client.generate(request)

    assert not result.success
    # 失败请求的 itl_list 应为默认空列表
    assert result.itl_list == []
    # 失败请求的 e2e_latency_ms 应为默认 0.0
    assert result.e2e_latency_ms == 0.0


@pytest.mark.asyncio
async def test_simulated_itl_in_metrics() -> None:
    """Test that ITL list is also stored in Protocol Metrics."""
    client = StubClient(ttft_ms=5.0, tbt_ms=2.0)

    request = BenchmarkRequest(
        prompt="Test",
        max_tokens=10,
        request_id="metrics-itl-test",
    )

    result = await client.generate(request)

    assert result.success
    assert result.metrics is not None

    # Protocol Metrics 中也应该有 itl_list
    assert result.metrics.itl_list is not None
    assert len(result.metrics.itl_list) == 10

    # BenchmarkResult 和 Metrics 中的 itl_list 应该相同
    assert result.itl_list == result.metrics.itl_list


@pytest.mark.asyncio
async def test_gateway_client_counts_real_tokens_not_stream_chunks(monkeypatch) -> None:
    class _FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [index for index, char in enumerate(text) if not char.isspace()]

    class _FakeChunk:
        def __init__(self, content: str) -> None:
            self.choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]

    class _FakeStream:
        def __init__(self, chunks: list[str]) -> None:
            self._chunks = chunks
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._chunks):
                raise StopAsyncIteration
            chunk = _FakeChunk(self._chunks[self._index])
            self._index += 1
            return chunk

    class _FakeCompletions:
        async def create(self, **kwargs):
            del kwargs
            return _FakeStream(["He", "llo"])

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    perf_times = iter([10.0, 10.05, 10.08, 10.10])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeTokenizer())
        ),
    )
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.time.perf_counter",
        lambda: next(perf_times),
    )

    client = GatewayClient(base_url="http://127.0.0.1:8000/v1")
    request = BenchmarkRequest(
        prompt="Hi all",
        max_tokens=8,
        request_id="gateway-001",
        model="/tmp/local-model",
    )

    result = await client.generate(request)

    assert result.success is True
    assert result.output_text == "Hello"
    assert result.output_tokens == 5
    assert result.prompt_tokens == 5
    assert result.metrics is not None
    assert result.metrics.ttft_ms == pytest.approx(50.0)
    assert result.metrics.tbt_ms == pytest.approx(12.5)
    assert result.metrics.tpot_ms == pytest.approx(20.0)
    assert result.metrics.throughput_tps == pytest.approx(50.0)


def test_gateway_client_prefers_explicit_local_model_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_LOCAL_MODEL_DIR", "/tmp/local-model-dir")
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.os.path.exists",
        lambda path: path == "/tmp/local-model-dir",
    )

    source, local_only = GatewayClient._resolve_tokenizer_source("Qwen/Qwen2.5-1.5B-Instruct")

    assert source == "/tmp/local-model-dir"
    assert local_only is True


def test_gateway_client_falls_back_to_local_hf_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    home = Path("/home/tester")
    monkeypatch.delenv("VLLM_LOCAL_MODEL_DIR", raising=False)
    monkeypatch.delenv("SAGELLM_BENCHMARK_LOCAL_MODEL_DIR", raising=False)
    monkeypatch.delenv("HF_LOCAL_MODEL_DIR", raising=False)
    monkeypatch.setattr("sagellm_benchmark.clients.openai_client.Path.home", lambda: home)

    expected = home / ".cache" / "hf-local-models" / "Qwen2.5-1.5B-Instruct"
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.Path.exists",
        lambda self: self == expected,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.os.path.exists",
        lambda path: False,
    )

    source, local_only = GatewayClient._resolve_tokenizer_source("Qwen/Qwen2.5-1.5B-Instruct")

    assert source == str(expected)
    assert local_only is True


def test_gateway_client_remote_tokenizer_defaults_to_hf_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return [1]

    def _from_pretrained(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["hf_endpoint"] = os.environ.get("HF_ENDPOINT")
        return _FakeTokenizer()

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.Path.home", lambda: Path("/no-local-cache")
    )
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.Path.exists",
        lambda self: False,
    )
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.os.path.exists",
        lambda path: False,
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=_from_pretrained)),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=lambda **kwargs: SimpleNamespace()),
    )

    client = GatewayClient(base_url="http://127.0.0.1:8000/v1")
    tokenizer = client._get_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

    assert tokenizer is not None
    assert captured["args"] == ("Qwen/Qwen2.5-1.5B-Instruct",)
    assert captured["kwargs"] == {"trust_remote_code": True, "local_files_only": False}
    assert captured["hf_endpoint"] == "https://hf-mirror.com"
