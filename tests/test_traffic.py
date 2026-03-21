"""测试流量控制模块 - Traffic Control Tests.

测试 ArrivalPattern, TrafficProfile, RequestGenerator, TrafficController 的功能。
"""

from __future__ import annotations

import statistics

import pytest
from test_helpers import StubClient

from sagellm_benchmark.traffic import (
    ArrivalPattern,
    RampUpStrategy,
    RequestGenerator,
    TrafficController,
    TrafficProfile,
)
from sagellm_benchmark.types import BenchmarkRequest

# ============================================================================
# Test ArrivalPattern Enum
# ============================================================================


def test_arrival_pattern_enum_values():
    """测试 ArrivalPattern 枚举值."""
    assert ArrivalPattern.INSTANT.value == "instant"
    assert ArrivalPattern.FIXED.value == "fixed"
    assert ArrivalPattern.POISSON.value == "poisson"
    assert ArrivalPattern.GAMMA.value == "gamma"


def test_arrival_pattern_enum_members():
    """测试 ArrivalPattern 枚举成员."""
    patterns = list(ArrivalPattern)
    assert len(patterns) == 5
    assert ArrivalPattern.INSTANT in patterns
    assert ArrivalPattern.FIXED in patterns
    assert ArrivalPattern.POISSON in patterns
    assert ArrivalPattern.GAMMA in patterns
    assert ArrivalPattern.BATCH in patterns


# ============================================================================
# Test TrafficProfile
# ============================================================================


def test_traffic_profile_defaults():
    """测试 TrafficProfile 默认值."""
    profile = TrafficProfile()
    assert profile.pattern == ArrivalPattern.INSTANT
    assert profile.request_rate is None
    assert profile.burstiness == 1.0
    assert profile.duration_s is None
    assert profile.warmup_requests == 0
    assert profile.num_prompts is None
    assert profile.ramp_up_strategy == RampUpStrategy.NONE
    assert profile.ramp_up_requests == 0
    assert profile.ramp_up_start_factor == 0.1
    assert profile.seed is None


def test_traffic_profile_custom_values():
    """测试 TrafficProfile 自定义值."""
    profile = TrafficProfile(
        pattern=ArrivalPattern.POISSON,
        request_rate=10.0,
        burstiness=0.5,
        duration_s=60.0,
        warmup_requests=5,
        num_prompts=100,
        ramp_up_strategy=RampUpStrategy.LINEAR,
        ramp_up_requests=20,
        ramp_up_start_factor=0.2,
        seed=42,
    )
    assert profile.pattern == ArrivalPattern.POISSON
    assert profile.request_rate == 10.0
    assert profile.burstiness == 0.5
    assert profile.duration_s == 60.0
    assert profile.warmup_requests == 5
    assert profile.num_prompts == 100
    assert profile.ramp_up_strategy == RampUpStrategy.LINEAR
    assert profile.ramp_up_requests == 20
    assert profile.ramp_up_start_factor == 0.2
    assert profile.seed == 42


def test_traffic_profile_rate_normalization():
    """测试 request_rate 归一化语义."""
    assert TrafficProfile(request_rate=None).normalized_request_rate is None
    assert TrafficProfile(request_rate=0.0).normalized_request_rate is None
    assert TrafficProfile(request_rate=-3.0).normalized_request_rate is None
    assert TrafficProfile(request_rate=float("inf")).normalized_request_rate is None
    assert TrafficProfile(request_rate=12.5).normalized_request_rate == 12.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"request_rate": float("nan")}, "request_rate"),
        ({"burstiness": 0.0}, "burstiness"),
        ({"warmup_requests": -1}, "warmup_requests"),
        ({"num_prompts": -1}, "num_prompts"),
        ({"ramp_up_requests": -1}, "ramp_up_requests"),
        ({"ramp_up_start_factor": 0.0}, "ramp_up_start_factor"),
    ],
)
def test_traffic_profile_invalid_values(kwargs: dict[str, float | int], message: str):
    """测试 TrafficProfile 边界校验."""
    with pytest.raises(ValueError, match=message):
        TrafficProfile(**kwargs)


# ============================================================================
# Test RequestGenerator
# ============================================================================


def _create_test_requests(count: int) -> list[BenchmarkRequest]:
    """创建测试请求列表."""
    return [
        BenchmarkRequest(
            prompt=f"Test prompt {i}",
            max_tokens=10,
            request_id=f"req-{i}",
            model="test-model",
            stream=False,
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_request_generator_instant_mode():
    """测试 INSTANT 模式：所有 delay 应该为 0."""
    requests = _create_test_requests(5)
    profile = TrafficProfile(pattern=ArrivalPattern.INSTANT)
    generator = RequestGenerator(requests, profile)

    delays = []
    request_ids = []
    async for delay, request in generator:
        delays.append(delay)
        request_ids.append(request.request_id)

    assert len(delays) == 5
    assert all(d == 0.0 for d in delays), "INSTANT mode should have zero delays"
    assert request_ids == ["req-0", "req-1", "req-2", "req-3", "req-4"]


@pytest.mark.asyncio
async def test_request_generator_fixed_mode():
    """测试 FIXED 模式：固定间隔（第一个为 0，后续固定）."""
    requests = _create_test_requests(5)
    profile = TrafficProfile(
        pattern=ArrivalPattern.FIXED,
        request_rate=10.0,  # 10 QPS → 0.1s 间隔
        seed=42,
    )
    generator = RequestGenerator(requests, profile)

    delays = []
    async for delay, request in generator:
        delays.append(delay)

    assert len(delays) == 5
    assert delays[0] == 0.0, "First request should have zero delay"
    for d in delays[1:]:
        assert abs(d - 0.1) < 1e-9, f"Expected 0.1s interval, got {d}"


@pytest.mark.asyncio
async def test_request_generator_poisson_mode():
    """测试 POISSON 模式：延迟应该非零且随机."""
    requests = _create_test_requests(10)
    profile = TrafficProfile(
        pattern=ArrivalPattern.POISSON,
        request_rate=10.0,  # 10 QPS → 平均 0.1s 间隔
        seed=42,
    )
    generator = RequestGenerator(requests, profile)

    delays = []
    async for delay, request in generator:
        delays.append(delay)

    assert len(delays) == 10
    # 泊松分布应该有非零延迟（几乎所有）
    non_zero_delays = [d for d in delays if d > 0]
    assert len(non_zero_delays) >= 8, "Most delays should be non-zero in POISSON mode"

    # 平均延迟应该接近 0.1s
    avg_delay = sum(delays) / len(delays)
    assert 0.05 < avg_delay < 0.15, f"Average delay {avg_delay} should be around 0.1s"


@pytest.mark.asyncio
async def test_request_generator_gamma_mode():
    """测试 GAMMA 模式：延迟应该非零且随机."""
    requests = _create_test_requests(10)
    profile = TrafficProfile(
        pattern=ArrivalPattern.GAMMA,
        request_rate=10.0,  # 10 QPS → 平均 0.1s 间隔
        burstiness=2.0,  # shape > 1 更均匀
        seed=42,
    )
    generator = RequestGenerator(requests, profile)

    delays = []
    async for delay, request in generator:
        delays.append(delay)

    assert len(delays) == 10
    # Gamma 分布应该有非零延迟
    non_zero_delays = [d for d in delays if d > 0]
    assert len(non_zero_delays) >= 8, "Most delays should be non-zero in GAMMA mode"

    # 平均延迟应该接近 0.1s
    avg_delay = sum(delays) / len(delays)
    assert 0.05 < avg_delay < 0.15, f"Average delay {avg_delay} should be around 0.1s"


@pytest.mark.asyncio
async def test_request_generator_no_rate_limit():
    """测试无速率限制：delay 应该为 0."""
    requests = _create_test_requests(5)
    profile = TrafficProfile(
        pattern=ArrivalPattern.POISSON,
        request_rate=None,  # 无速率限制
    )
    generator = RequestGenerator(requests, profile)

    delays = []
    async for delay, request in generator:
        delays.append(delay)

    assert all(d == 0.0 for d in delays), "No rate limit should result in zero delays"


@pytest.mark.asyncio
async def test_request_generator_zero_rate():
    """测试零速率：delay 应该为 0."""
    requests = _create_test_requests(5)
    profile = TrafficProfile(
        pattern=ArrivalPattern.POISSON,
        request_rate=0.0,  # 零速率
    )
    generator = RequestGenerator(requests, profile)

    delays = []
    async for delay, request in generator:
        delays.append(delay)

    assert all(d == 0.0 for d in delays), "Zero rate should result in zero delays"


@pytest.mark.asyncio
async def test_request_generator_infinite_rate_normalized_to_unlimited():
    """测试无限速率会归一化为不限速."""
    requests = _create_test_requests(5)
    profile = TrafficProfile(
        pattern=ArrivalPattern.GAMMA,
        request_rate=float("inf"),
        burstiness=0.5,
    )
    generator = RequestGenerator(requests, profile)

    delays = [delay async for delay, _ in generator]

    assert delays == [0.0] * 5


@pytest.mark.asyncio
async def test_request_generator_reproducibility():
    """测试随机种子可复现性."""
    requests = _create_test_requests(5)

    # 第一次运行
    profile1 = TrafficProfile(
        pattern=ArrivalPattern.POISSON,
        request_rate=10.0,
        seed=42,
    )
    generator1 = RequestGenerator(requests, profile1)
    delays1 = [delay async for delay, _ in generator1]

    # 第二次运行（相同种子）
    profile2 = TrafficProfile(
        pattern=ArrivalPattern.POISSON,
        request_rate=10.0,
        seed=42,
    )
    generator2 = RequestGenerator(requests, profile2)
    delays2 = [delay async for delay, _ in generator2]

    # 应该完全相同
    assert delays1 == delays2, "Same seed should produce same delays"


@pytest.mark.asyncio
async def test_request_generator_gamma_burstiness_impacts_variance():
    """测试 burstiness 会改变 Gamma 分布离散程度，但保持目标速率均值."""
    requests = _create_test_requests(2000)
    low_burst_profile = TrafficProfile(
        pattern=ArrivalPattern.GAMMA,
        request_rate=10.0,
        burstiness=0.5,
        seed=7,
    )
    high_burst_profile = TrafficProfile(
        pattern=ArrivalPattern.GAMMA,
        request_rate=10.0,
        burstiness=5.0,
        seed=7,
    )

    low_burst_delays = [delay async for delay, _ in RequestGenerator(requests, low_burst_profile)]
    high_burst_delays = [delay async for delay, _ in RequestGenerator(requests, high_burst_profile)]

    low_burst_std = statistics.pstdev(low_burst_delays)
    high_burst_std = statistics.pstdev(high_burst_delays)
    low_burst_avg = sum(low_burst_delays) / len(low_burst_delays)
    high_burst_avg = sum(high_burst_delays) / len(high_burst_delays)

    assert low_burst_std > high_burst_std
    assert low_burst_avg == pytest.approx(0.1, rel=0.12)
    assert high_burst_avg == pytest.approx(0.1, rel=0.08)


@pytest.mark.asyncio
async def test_request_generator_linear_ramp_up():
    """测试 linear ramp-up 会逐步逼近目标 request_rate."""
    requests = _create_test_requests(6)
    profile = TrafficProfile(
        pattern=ArrivalPattern.FIXED,
        request_rate=10.0,
        ramp_up_strategy=RampUpStrategy.LINEAR,
        ramp_up_requests=4,
        ramp_up_start_factor=0.25,
    )

    delays = [delay async for delay, _ in RequestGenerator(requests, profile)]

    assert delays[0] == 0.0
    assert delays[1:] == pytest.approx([0.4, 0.2, 0.1333333333, 0.1, 0.1], rel=1e-6)


@pytest.mark.asyncio
async def test_request_generator_exponential_ramp_up():
    """测试 exponential ramp-up 前期更保守，后期快速逼近目标 request_rate."""
    requests = _create_test_requests(6)
    profile = TrafficProfile(
        pattern=ArrivalPattern.FIXED,
        request_rate=10.0,
        ramp_up_strategy=RampUpStrategy.EXPONENTIAL,
        ramp_up_requests=4,
        ramp_up_start_factor=0.125,
    )

    delays = [delay async for delay, _ in RequestGenerator(requests, profile)]

    assert delays[0] == 0.0
    assert delays[1:] == pytest.approx([0.8, 0.4, 0.2, 0.1, 0.1], rel=1e-6)


# ============================================================================
# Test TrafficController
# ============================================================================


@pytest.mark.asyncio
async def test_traffic_controller_instant_mode():
    """测试 TrafficController INSTANT 模式."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(pattern=ArrivalPattern.INSTANT)
    controller = TrafficController(client, profile)

    requests = _create_test_requests(5)
    results = await controller.run(requests)

    assert len(results) == 5
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_traffic_controller_fixed_mode():
    """测试 TrafficController FIXED 模式."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.FIXED,
        request_rate=100.0,  # 100 QPS → 0.01s 间隔（快速测试）
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(3)
    results = await controller.run(requests)

    assert len(results) == 3
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_traffic_controller_warmup():
    """测试 TrafficController warmup 机制."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.INSTANT,
        warmup_requests=3,
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(10)  # 10 个请求
    results = await controller.run(requests)

    # 前 3 个是 warmup，应该返回剩余 7 个结果
    assert len(results) == 7
    assert all(r.success for r in results)
    # 验证返回的是后 7 个请求
    expected_ids = [f"req-{i}" for i in range(3, 10)]
    actual_ids = [r.request_id for r in results]
    assert actual_ids == expected_ids


@pytest.mark.asyncio
async def test_traffic_controller_warmup_exceeds_requests():
    """测试 warmup 数量超过请求总数."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.INSTANT,
        warmup_requests=10,  # warmup 10 个
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(5)  # 只有 5 个请求
    results = await controller.run(requests)

    # 所有请求都用于 warmup，正式测试应该返回空列表
    assert len(results) == 0


@pytest.mark.asyncio
async def test_traffic_controller_no_warmup():
    """测试 TrafficController 无 warmup."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.INSTANT,
        warmup_requests=0,
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(5)
    results = await controller.run(requests)

    assert len(results) == 5
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_traffic_controller_num_prompts_limit_applies_after_warmup():
    """测试 num_prompts 只限制正式测试阶段请求数."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.INSTANT,
        warmup_requests=2,
        num_prompts=3,
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(10)
    results = await controller.run(requests)

    assert len(results) == 3
    assert [result.request_id for result in results] == ["req-2", "req-3", "req-4"]


@pytest.mark.asyncio
async def test_traffic_controller_zero_num_prompts_returns_empty_results():
    """测试 num_prompts=0 时正式测试阶段直接返回空结果."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.INSTANT,
        num_prompts=0,
    )
    controller = TrafficController(client, profile)

    results = await controller.run(_create_test_requests(5))

    assert results == []


@pytest.mark.asyncio
async def test_traffic_controller_poisson_mode():
    """测试 TrafficController POISSON 模式."""
    client = StubClient(ttft_ms=5.0, tbt_ms=2.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.POISSON,
        request_rate=50.0,  # 50 QPS（快速测试）
        seed=42,
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(5)
    results = await controller.run(requests)

    assert len(results) == 5
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_traffic_controller_empty_requests():
    """测试 TrafficController 空请求列表."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(pattern=ArrivalPattern.INSTANT)
    controller = TrafficController(client, profile)

    results = await controller.run([])

    assert len(results) == 0


@pytest.mark.asyncio
async def test_traffic_controller_gamma_mode():
    """测试 TrafficController GAMMA 模式."""
    client = StubClient(ttft_ms=5.0, tbt_ms=2.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.GAMMA,
        request_rate=50.0,  # 50 QPS
        burstiness=1.5,
        seed=42,
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(5)
    results = await controller.run(requests)

    assert len(results) == 5
    assert all(r.success for r in results)


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_traffic_controller_integration():
    """集成测试：完整流程."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)

    # 配置：POISSON 模式，10 QPS，2 个 warmup
    profile = TrafficProfile(
        pattern=ArrivalPattern.POISSON,
        request_rate=100.0,  # 快速测试
        warmup_requests=2,
        seed=42,
    )

    controller = TrafficController(client, profile)
    requests = _create_test_requests(10)

    results = await controller.run(requests)

    # 验证结果
    assert len(results) == 8, "Should have 8 results (10 - 2 warmup)"
    assert all(r.success for r in results), "All requests should succeed"

    # 验证请求 ID
    expected_ids = [f"req-{i}" for i in range(2, 10)]
    actual_ids = [r.request_id for r in results]
    assert actual_ids == expected_ids


@pytest.mark.asyncio
async def test_request_order_preserved():
    """测试请求顺序保持一致."""
    client = StubClient(ttft_ms=5.0, tbt_ms=2.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.FIXED,
        request_rate=100.0,
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(10)
    results = await controller.run(requests)

    # 验证顺序
    expected_ids = [f"req-{i}" for i in range(10)]
    actual_ids = [r.request_id for r in results]
    assert actual_ids == expected_ids, "Request order should be preserved"


# ============================================================================
# Test BATCH Mode
# ============================================================================


@pytest.mark.asyncio
async def test_arrival_pattern_batch():
    """测试 BATCH 模式枚举值."""
    assert ArrivalPattern.BATCH.value == "batch"


@pytest.mark.asyncio
async def test_request_generator_batch_mode():
    """测试 BATCH 模式：所有 delay 应该为 0（类似 INSTANT）."""
    requests = _create_test_requests(5)
    profile = TrafficProfile(pattern=ArrivalPattern.BATCH)
    generator = RequestGenerator(requests, profile)

    delays = []
    request_ids = []
    async for delay, request in generator:
        delays.append(delay)
        request_ids.append(request.request_id)

    assert len(delays) == 5
    assert all(d == 0.0 for d in delays), "BATCH mode should have zero delays"
    assert request_ids == ["req-0", "req-1", "req-2", "req-3", "req-4"]


@pytest.mark.asyncio
async def test_traffic_controller_batch_mode():
    """测试 TrafficController BATCH 模式：并发执行 + 总时长统计."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.BATCH,
        enable_batch_mode=True,
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(5)
    results = await controller.run(requests)

    assert len(results) == 5
    assert all(r.success for r in results)

    # 验证 batch 模式下所有结果都有 _batch_total_time_s 属性
    assert all(hasattr(r, "_batch_total_time_s") for r in results), (
        "BATCH mode should add _batch_total_time_s to results"
    )

    # 验证总时长是合理的（应该 > 0）
    total_time = results[0]._batch_total_time_s
    assert total_time > 0, "Batch total time should be positive"


@pytest.mark.asyncio
async def test_traffic_controller_batch_with_warmup():
    """测试 TrafficController BATCH 模式 + warmup."""
    client = StubClient(ttft_ms=10.0, tbt_ms=5.0)
    profile = TrafficProfile(
        pattern=ArrivalPattern.BATCH,
        enable_batch_mode=True,
        warmup_requests=3,
    )
    controller = TrafficController(client, profile)

    requests = _create_test_requests(10)  # 10 个请求
    results = await controller.run(requests)

    # 前 3 个是 warmup，应该返回剩余 7 个结果
    assert len(results) == 7
    assert all(r.success for r in results)

    # 验证返回的是后 7 个请求
    expected_ids = [f"req-{i}" for i in range(3, 10)]
    actual_ids = [r.request_id for r in results]
    assert actual_ids == expected_ids

    # 验证 batch 统计
    assert all(hasattr(r, "_batch_total_time_s") for r in results)


@pytest.mark.asyncio
async def test_traffic_profile_batch_mode():
    """测试 TrafficProfile enable_batch_mode 属性."""
    profile = TrafficProfile(
        pattern=ArrivalPattern.BATCH,
        enable_batch_mode=True,
        warmup_requests=5,
    )

    assert profile.pattern == ArrivalPattern.BATCH
    assert profile.enable_batch_mode is True
    assert profile.warmup_requests == 5
