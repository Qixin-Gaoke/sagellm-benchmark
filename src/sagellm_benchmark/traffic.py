"""流量控制模块 - Traffic Control Layer.

此模块提供多种请求到达模式和流量控制功能：
- ArrivalPattern: 请求到达模式枚举（INSTANT/FIXED/POISSON/GAMMA）
- TrafficProfile: 流量配置数据类
- RequestGenerator: 请求发生器（生成带延迟的请求序列）
- TrafficController: 流量控制器（封装完整压测流程）

参考 vLLM 的请求发生器设计，但保持简洁独立。
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sagellm_benchmark.clients.base import BenchmarkClient
    from sagellm_benchmark.types import BenchmarkRequest, BenchmarkResult

logger = logging.getLogger(__name__)


class ArrivalPattern(StrEnum):
    """请求到达模式枚举.

    Attributes:
        INSTANT: 立即发送所有请求（现有行为，兼容旧代码）
        FIXED: 固定间隔发送
        POISSON: 泊松分布（指数间隔）
        GAMMA: Gamma 分布（支持突发流量控制）
        BATCH: 本地批量提交模式（一次性提交所有请求，测量总耗时）
    """

    INSTANT = "instant"
    FIXED = "fixed"
    POISSON = "poisson"
    GAMMA = "gamma"
    BATCH = "batch"


class RampUpStrategy(StrEnum):
    """请求速率爬升策略.

    Attributes:
        NONE: 不启用爬升，直接使用目标 request_rate。
        LINEAR: 在 ramp-up 窗口内按线性曲线从低速爬升到目标速率。
        EXPONENTIAL: 在 ramp-up 窗口内按指数曲线从低速爬升到目标速率。
    """

    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class TrafficProfile:
    """流量配置数据类.

    Attributes:
        pattern: 到达模式，默认 INSTANT（兼容现有行为）。
        request_rate: 请求速率（请求/秒），None 表示不限速。
        burstiness: Gamma 分布形状参数，1.0 = 泊松，<1 更突发，>1 更均匀。
        duration_s: 持续时间（秒），None 表示发完所有请求即停止。
        warmup_requests: 预热请求数（不计入统计）。
        num_prompts: 正式测试阶段最多发射多少请求，None 表示使用全部剩余请求。
        ramp_up_strategy: request_rate 爬升策略。
        ramp_up_requests: 进入目标 request_rate 前的爬升窗口大小。
        ramp_up_start_factor: 爬升起点相对目标 request_rate 的比例，范围 (0, 1]。
        seed: 随机种子（用于可复现测试）。
        enable_batch_mode: 是否启用 Batch 模式（本地一次性提交请求并测量总耗时）。

    Example:
        >>> # 泊松分布，10 QPS，5 个预热请求
        >>> profile = TrafficProfile(
        ...     pattern=ArrivalPattern.POISSON,
        ...     request_rate=10.0,
        ...     warmup_requests=5,
        ...     seed=42,
        ... )
        >>> # 固定间隔，20 QPS
        >>> profile = TrafficProfile(
        ...     pattern=ArrivalPattern.FIXED,
        ...     request_rate=20.0,
        ... )
        >>> # 本地 Batch 模式
        >>> profile = TrafficProfile(
        ...     pattern=ArrivalPattern.BATCH,
        ...     enable_batch_mode=True,
        ...     warmup_requests=10,
        ... )
    """

    pattern: ArrivalPattern = ArrivalPattern.INSTANT
    request_rate: float | None = None
    burstiness: float = 1.0
    duration_s: float | None = None
    warmup_requests: int = 0
    num_prompts: int | None = None
    ramp_up_strategy: RampUpStrategy = RampUpStrategy.NONE
    ramp_up_requests: int = 0
    ramp_up_start_factor: float = 0.1
    seed: int | None = None
    enable_batch_mode: bool = False

    def __post_init__(self) -> None:
        """校验并规范化配置."""
        if self.request_rate is not None and math.isnan(self.request_rate):
            raise ValueError("request_rate cannot be NaN")
        if self.burstiness <= 0:
            raise ValueError("burstiness must be > 0")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be >= 0")
        if self.num_prompts is not None and self.num_prompts < 0:
            raise ValueError("num_prompts must be >= 0")
        if self.ramp_up_requests < 0:
            raise ValueError("ramp_up_requests must be >= 0")
        if not 0 < self.ramp_up_start_factor <= 1.0:
            raise ValueError("ramp_up_start_factor must be in (0, 1]")

    @property
    def normalized_request_rate(self) -> float | None:
        """返回归一化后的请求速率.

        `None`、非正数和 `inf` 都视为不限速，以便对齐 vLLM 常见
        `--request-rate inf` 语义，同时保持现有零延迟行为。
        """
        if self.request_rate is None:
            return None
        if math.isinf(self.request_rate):
            return None
        if self.request_rate <= 0:
            return None
        return self.request_rate

    def limit_actual_requests(
        self,
        requests: list[BenchmarkRequest],
    ) -> list[BenchmarkRequest]:
        """限制正式测试阶段的请求数量."""
        if self.num_prompts is None:
            return requests
        return requests[: self.num_prompts]


class RequestGenerator:
    """请求发生器 - 根据 TrafficProfile 控制请求发射节奏.

    根据配置的流量模式生成带延迟的请求序列。支持：
    - INSTANT: 所有请求延迟为 0（并发执行）
    - FIXED: 固定间隔
    - POISSON: 泊松分布（指数间隔）
    - GAMMA: Gamma 分布（可控突发）

    Example:
        >>> profile = TrafficProfile(
        ...     pattern=ArrivalPattern.POISSON,
        ...     request_rate=10.0,
        ...     seed=42,
        ... )
        >>> generator = RequestGenerator(requests, profile)
        >>> async for delay, request in generator:
        ...     await asyncio.sleep(delay)
        ...     result = await client.generate(request)
    """

    def __init__(
        self,
        requests: list[BenchmarkRequest],
        profile: TrafficProfile,
    ) -> None:
        """初始化请求发生器.

        Args:
            requests: 要发送的请求列表。
            profile: 流量配置。
        """
        self.requests = requests
        self.profile = profile
        self._rng = random.Random(profile.seed)
        logger.debug(
            f"RequestGenerator initialized: pattern={profile.pattern.value}, "
            f"rate={profile.request_rate}, requests={len(requests)}, "
            f"ramp={profile.ramp_up_strategy.value}"
        )

    def __aiter__(self) -> AsyncIterator[tuple[float, BenchmarkRequest]]:
        """返回异步迭代器."""
        return self._generate()

    async def _generate(self) -> AsyncIterator[tuple[float, BenchmarkRequest]]:
        """生成 (delay_seconds, request) 序列.

        Yields:
            tuple[float, BenchmarkRequest]: (延迟秒数, 请求对象)
        """
        for i, request in enumerate(self.requests):
            delay = self._compute_delay(i)
            yield delay, request

    def _compute_delay(self, index: int) -> float:
        """根据模式计算下一个请求的延迟.

        Args:
            index: 请求序号（从 0 开始）。

        Returns:
            延迟秒数（>= 0）。

        Note:
            - INSTANT 模式：所有延迟为 0
            - BATCH 模式：所有延迟为 0（类似 INSTANT，但在 TrafficController 中有特殊处理）
            - FIXED 模式：第一个请求延迟 0，后续请求固定间隔
            - POISSON 模式：使用指数分布
            - GAMMA 模式：使用 Gamma 分布
        """
        # INSTANT 或 BATCH 模式：无延迟
        if self.profile.pattern in (ArrivalPattern.INSTANT, ArrivalPattern.BATCH):
            return 0.0

        # 未设置速率：无延迟
        request_rate = self._effective_request_rate(index)
        if request_rate is None:
            return 0.0

        mean_interval = 1.0 / request_rate

        # FIXED 模式：固定间隔（第一个请求无延迟）
        if self.profile.pattern == ArrivalPattern.FIXED:
            return mean_interval if index > 0 else 0.0

        # POISSON 模式：指数分布
        elif self.profile.pattern == ArrivalPattern.POISSON:
            return self._rng.expovariate(request_rate)

        # GAMMA 模式：Gamma 分布
        elif self.profile.pattern == ArrivalPattern.GAMMA:
            shape = self.profile.burstiness
            scale = mean_interval / shape
            return self._rng.gammavariate(shape, scale)

        # 未知模式：无延迟
        return 0.0

    def _effective_request_rate(self, index: int) -> float | None:
        """计算当前请求的有效 request_rate."""
        base_rate = self.profile.normalized_request_rate
        if base_rate is None:
            return None
        return base_rate * self._ramp_up_factor(index)

    def _ramp_up_factor(self, index: int) -> float:
        """根据 ramp-up 配置计算速率缩放因子."""
        strategy = self.profile.ramp_up_strategy
        ramp_up_requests = self.profile.ramp_up_requests
        if strategy == RampUpStrategy.NONE or ramp_up_requests <= 0:
            return 1.0

        ramp_step = max(index - 1, 0)
        if ramp_step >= ramp_up_requests:
            return 1.0

        start_factor = self.profile.ramp_up_start_factor
        if ramp_up_requests == 1:
            return start_factor

        progress = ramp_step / (ramp_up_requests - 1)
        if strategy == RampUpStrategy.LINEAR:
            return start_factor + ((1.0 - start_factor) * progress)
        if strategy == RampUpStrategy.EXPONENTIAL:
            return start_factor * ((1.0 / start_factor) ** progress)
        return 1.0


class TrafficController:
    """流量控制器 - 封装完整的压测流程.

    封装 warmup → 正式测试 → 结果收集 的完整流程。
    支持所有 ArrivalPattern 模式和 warmup 机制。

    Attributes:
        client: BenchmarkClient 实例。
        profile: TrafficProfile 配置。

    Example:
        >>> profile = TrafficProfile(
        ...     pattern=ArrivalPattern.POISSON,
        ...     request_rate=10.0,
        ...     warmup_requests=5,
        ... )
        >>> controller = TrafficController(client, profile)
        >>> results = await controller.run(requests)
        >>> # results 不包含 warmup 请求的结果
    """

    def __init__(
        self,
        client: BenchmarkClient,
        profile: TrafficProfile,
    ) -> None:
        """初始化流量控制器.

        Args:
            client: BenchmarkClient 实例。
            profile: TrafficProfile 配置。
        """
        self.client = client
        self.profile = profile
        logger.info(
            f"TrafficController initialized: client={client.name}, "
            f"pattern={profile.pattern.value}, warmup={profile.warmup_requests}"
        )

    async def run(
        self,
        requests: list[BenchmarkRequest],
    ) -> list[BenchmarkResult]:
        """执行压测流程.

        流程：
        1. 如果配置了 warmup_requests，先执行预热（结果丢弃）
        2. 执行正式测试
        3. 返回正式测试结果

        Args:
            requests: 要执行的请求列表（包含 warmup + 正式测试）。

        Returns:
            正式测试的结果列表（不含 warmup 结果）。

        Note:
            - warmup 请求从 requests 前部提取
            - 正式测试请求为剩余请求
            - 如果 requests 数量少于 warmup_requests，则全部用于 warmup，正式测试返回空列表
            - BATCH 模式会记录总耗时并添加到每个结果的 e2e_latency_ms 中
        """
        all_requests = requests.copy()

        # Warmup 阶段
        warmup_count = self.profile.warmup_requests
        if warmup_count > 0 and len(all_requests) > 0:
            # 提取 warmup 请求
            actual_warmup_count = min(warmup_count, len(all_requests))
            warmup_reqs = all_requests[:actual_warmup_count]
            all_requests = all_requests[actual_warmup_count:]

            logger.info(f"Starting warmup phase: {len(warmup_reqs)} requests")
            await self._run_requests(warmup_reqs, is_warmup=True)
            logger.info("Warmup phase completed")

        # 正式测试
        if not all_requests:
            logger.warning("No requests left for actual testing after warmup")
            return []

        all_requests = self.profile.limit_actual_requests(all_requests)
        if not all_requests:
            logger.warning("No requests left for actual testing after num_prompts limit")
            return []

        logger.info(f"Starting actual test phase: {len(all_requests)} requests")

        # Batch 模式：记录总耗时
        if self.profile.pattern == ArrivalPattern.BATCH or self.profile.enable_batch_mode:
            import time

            start_time = time.perf_counter()
            results = await self._run_requests(all_requests, is_warmup=False)
            end_time = time.perf_counter()
            total_time_s = end_time - start_time

            # 为每个结果添加总耗时信息（用于后续聚合指标计算）
            # 注意：这里的 e2e_latency_ms 在 BATCH 模式下表示总时长，不是单个请求的延迟
            for result in results:
                result.batch_total_time_s = total_time_s
                if not hasattr(result, "_batch_total_time_s"):
                    result._batch_total_time_s = total_time_s

            logger.info(
                f"Batch test completed: {len(results)} results, total_time={total_time_s:.3f}s"
            )
        else:
            results = await self._run_requests(all_requests, is_warmup=False)
            logger.info(f"Test phase completed: {len(results)} results")

        return results

    async def _run_requests(
        self,
        requests: list[BenchmarkRequest],
        is_warmup: bool,
    ) -> list[BenchmarkResult]:
        """按 profile 发射请求并收集结果.

        Args:
            requests: 要执行的请求列表。
            is_warmup: 是否为 warmup 阶段（仅用于日志）。

        Returns:
            结果列表（与 requests 顺序一致）。

        Note:
            - INSTANT 和 BATCH 模式：使用 asyncio.gather 并发执行
            - 其他模式：按延迟顺序执行（流式）
        """
        results: list[BenchmarkResult] = []
        generator = RequestGenerator(requests, self.profile)

        # INSTANT 或 BATCH 模式：并发执行
        if self.profile.pattern in (ArrivalPattern.INSTANT, ArrivalPattern.BATCH):
            tasks = []
            async for delay, request in generator:
                # INSTANT/BATCH 模式下 delay 应该为 0，但仍然尊重返回值
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(asyncio.create_task(self.client.generate(request)))

            # 等待所有任务完成
            if tasks:
                mode_name = "BATCH" if self.profile.pattern == ArrivalPattern.BATCH else "INSTANT"
                logger.debug(f"Running {len(tasks)} requests concurrently ({mode_name} mode)")
                results = await asyncio.gather(*tasks, return_exceptions=False)
                results = list(results)

        # 其他模式：流式执行
        else:
            async for delay, request in generator:
                if delay > 0:
                    await asyncio.sleep(delay)
                result = await self.client.generate(request)
                results.append(result)

        return results
