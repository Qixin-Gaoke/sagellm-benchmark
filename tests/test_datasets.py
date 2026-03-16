"""数据集模块单元测试。

测试 datasets 模块的核心功能：
- RandomDataset 采样
- ShareGPT 数据加载
- Year Demo Workloads
- 接口契约验证
"""

from __future__ import annotations

import uuid

import pytest

from sagellm_benchmark.datasets import (
    YEAR1_WORKLOADS,
    BenchmarkDataset,
    RandomDataset,
    SyntheticShareGPTDataset,
    build_serving_requests,
    create_custom_workload,
    get_workload_by_type,
    get_workloads_by_year,
    get_year1_workloads,
    get_year2_workloads,
    get_year3_workloads,
    load_serving_dataset,
)
from sagellm_benchmark.types import BenchmarkRequest, WorkloadSpec, WorkloadType


class TestWorkloadSpec:
    """WorkloadSpec 数据类测试。"""

    def test_create_short_workload(self) -> None:
        """测试创建 SHORT workload。"""
        spec = WorkloadSpec(
            name="test_short",
            workload_type=WorkloadType.SHORT,
            prompt_len=128,
            output_len=64,
            num_requests=5,
        )
        assert spec.name == "test_short"
        assert spec.workload_type == WorkloadType.SHORT
        assert spec.prompt_len == 128
        assert spec.output_len == 64
        assert spec.num_requests == 5
        assert spec.concurrent is False
        assert spec.kv_budget_tokens is None

    def test_create_stress_workload(self) -> None:
        """测试创建 STRESS workload。"""
        spec = WorkloadSpec(
            name="test_stress",
            workload_type=WorkloadType.STRESS,
            prompt_len=256,
            output_len=128,
            num_requests=10,
            concurrent=True,
            kv_budget_tokens=4096,
        )
        assert spec.concurrent is True
        assert spec.kv_budget_tokens == 4096


class TestBenchmarkRequest:
    """BenchmarkRequest 数据类测试。"""

    def test_create_request_minimal(self) -> None:
        """测试最小化创建请求。"""
        request = BenchmarkRequest(
            prompt="Hello world",
            max_tokens=64,
            request_id="test-123",
        )
        assert request.prompt == "Hello world"
        assert request.max_tokens == 64
        assert request.request_id == "test-123"
        # 默认值
        assert request.model == "default"
        assert request.stream is False
        assert request.temperature is None
        assert request.top_p is None
        assert request.kv_budget_tokens is None

    def test_create_request_full(self) -> None:
        """测试完整参数创建请求。"""
        request = BenchmarkRequest(
            prompt="Test prompt",
            max_tokens=128,
            request_id="full-test",
            model="llama-7b",
            stream=True,
            temperature=0.7,
            top_p=0.9,
            kv_budget_tokens=2048,
        )
        assert request.model == "llama-7b"
        assert request.stream is True
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.kv_budget_tokens == 2048


class TestRandomDataset:
    """RandomDataset 测试。"""

    def test_sample_basic(self) -> None:
        """测试基础采样功能。"""
        dataset = RandomDataset(seed=42)
        spec = WorkloadSpec(
            name="test",
            workload_type=WorkloadType.SHORT,
            prompt_len=128,
            output_len=64,
            num_requests=5,
        )

        requests = dataset.sample(spec)

        assert len(requests) == 5
        for req in requests:
            assert isinstance(req, BenchmarkRequest)
            assert req.max_tokens == 64
            assert len(req.prompt) > 0
            # 验证 request_id 是有效 UUID
            uuid.UUID(req.request_id)

    def test_sample_unique_ids(self) -> None:
        """测试 request_id 唯一性。"""
        dataset = RandomDataset(seed=42)
        spec = WorkloadSpec(
            name="test",
            workload_type=WorkloadType.SHORT,
            prompt_len=100,
            output_len=50,
            num_requests=10,
        )

        requests = dataset.sample(spec)
        request_ids = [r.request_id for r in requests]

        # 所有 ID 必须唯一
        assert len(request_ids) == len(set(request_ids))

    def test_sample_reproducible(self) -> None:
        """测试采样可复现性。"""
        spec = WorkloadSpec(
            name="test",
            workload_type=WorkloadType.SHORT,
            prompt_len=100,
            output_len=50,
            num_requests=3,
        )

        dataset1 = RandomDataset(seed=123)
        requests1 = dataset1.sample(spec)

        dataset2 = RandomDataset(seed=123)
        requests2 = dataset2.sample(spec)

        # 相同种子应产生相同 prompt
        for r1, r2 in zip(requests1, requests2):
            assert r1.prompt == r2.prompt

    def test_sample_prompt_length(self) -> None:
        """测试 prompt 长度控制。"""
        dataset = RandomDataset(seed=42, length_mode="token")
        spec = WorkloadSpec(
            name="test",
            workload_type=WorkloadType.LONG,
            prompt_len=500,  # 500 tokens
            output_len=100,
            num_requests=3,
        )

        requests = dataset.sample(spec)

        # 500 tokens ≈ 2000 chars，允许 ±20% 误差
        for req in requests:
            prompt_len = len(req.prompt)
            assert 1600 <= prompt_len <= 2400, f"Prompt length {prompt_len} out of range"

    def test_sample_with_kv_budget(self) -> None:
        """测试 KV 预算传递。"""
        dataset = RandomDataset(seed=42)
        spec = WorkloadSpec(
            name="test",
            workload_type=WorkloadType.STRESS,
            prompt_len=100,
            output_len=50,
            num_requests=3,
            kv_budget_tokens=2048,
        )

        requests = dataset.sample(spec)

        for req in requests:
            assert req.kv_budget_tokens == 2048

    def test_sample_invalid_spec(self) -> None:
        """测试无效规格参数。"""
        dataset = RandomDataset(seed=42)

        with pytest.raises(ValueError, match="prompt_len must be positive"):
            dataset.sample(
                WorkloadSpec(
                    name="invalid",
                    workload_type=WorkloadType.SHORT,
                    prompt_len=0,
                    output_len=50,
                    num_requests=1,
                )
            )

        with pytest.raises(ValueError, match="num_requests must be positive"):
            dataset.sample(
                WorkloadSpec(
                    name="invalid",
                    workload_type=WorkloadType.SHORT,
                    prompt_len=100,
                    output_len=50,
                    num_requests=-1,
                )
            )

    def test_reset_seed(self) -> None:
        """测试重置种子。"""
        dataset = RandomDataset(seed=42)
        spec = WorkloadSpec(
            name="test",
            workload_type=WorkloadType.SHORT,
            prompt_len=50,
            output_len=25,
            num_requests=2,
        )

        requests1 = dataset.sample(spec)
        dataset.reset_seed()
        requests2 = dataset.sample(spec)

        # 重置后应产生相同结果
        for r1, r2 in zip(requests1, requests2):
            assert r1.prompt == r2.prompt


class TestSyntheticShareGPTDataset:
    """SyntheticShareGPTDataset 测试。"""

    def test_sample_basic(self) -> None:
        """测试基础采样。"""
        dataset = SyntheticShareGPTDataset(seed=42)
        spec = WorkloadSpec(
            name="test",
            workload_type=WorkloadType.SHORT,
            prompt_len=100,
            output_len=50,
            num_requests=5,
        )

        requests = dataset.sample(spec)

        assert len(requests) == 5
        for req in requests:
            assert isinstance(req, BenchmarkRequest)
            assert "?" in req.prompt or len(req.prompt) > 0  # 问答形式

    def test_dataset_name(self) -> None:
        """测试数据集名称。"""
        dataset = SyntheticShareGPTDataset()
        assert dataset.name == "synthetic_sharegpt"


class TestServingDatasetHelpers:
    """Serving benchmark dataset helper 测试。"""

    def test_load_serving_dataset_random(self) -> None:
        dataset = load_serving_dataset("random", seed=7)
        assert isinstance(dataset, RandomDataset)
        assert dataset.name == "random"

    def test_load_serving_dataset_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="Unsupported serving dataset"):
            load_serving_dataset("unknown")

    def test_build_serving_requests_random_is_reproducible(self) -> None:
        requests1 = build_serving_requests(
            dataset_name="random",
            num_prompts=3,
            input_len=64,
            output_len=16,
            model="demo-model",
            stream=True,
            seed=42,
        )
        requests2 = build_serving_requests(
            dataset_name="random",
            num_prompts=3,
            input_len=64,
            output_len=16,
            model="demo-model",
            stream=True,
            seed=42,
        )

        assert [request.prompt for request in requests1] == [
            request.prompt for request in requests2
        ]
        assert [request.request_id for request in requests1] == ["random-0", "random-1", "random-2"]
        assert all(request.stream for request in requests1)
        assert all(request.model == "demo-model" for request in requests1)
        assert all(request.max_tokens == 16 for request in requests1)

    def test_build_serving_requests_sharegpt_uses_existing_loader(self, monkeypatch) -> None:
        sample_data = [
            {
                "conversations": [
                    {"from": "human", "value": "Explain tensor parallelism in two paragraphs."},
                    {"from": "gpt", "value": "Tensor parallelism splits tensors across devices."},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Describe continuous batching for LLM serving."},
                    {"from": "gpt", "value": "Continuous batching keeps the decode queue full."},
                ]
            },
        ]

        def fake_from_huggingface(**kwargs):
            return __import__(
                "sagellm_benchmark.datasets.sharegpt",
                fromlist=["ShareGPTDataset"],
            ).ShareGPTDataset(sample_data, seed=kwargs.get("seed"))

        monkeypatch.setattr(
            "sagellm_benchmark.datasets.serving.ShareGPTDataset.from_huggingface",
            fake_from_huggingface,
        )

        requests = build_serving_requests(
            dataset_name="sharegpt",
            num_prompts=2,
            input_len=8,
            output_len=24,
            model="sharegpt-model",
            stream=False,
            seed=11,
        )

        assert len(requests) == 2
        assert [request.request_id for request in requests] == ["sharegpt-0", "sharegpt-1"]
        assert all(request.model == "sharegpt-model" for request in requests)
        assert all(request.stream is False for request in requests)
        assert all(request.max_tokens == 24 for request in requests)

    @pytest.mark.parametrize(
        ("num_prompts", "input_len", "output_len", "message"),
        [
            (0, 64, 16, "num_prompts"),
            (2, 0, 16, "input_len"),
            (2, 64, 0, "output_len"),
        ],
    )
    def test_build_serving_requests_rejects_invalid_lengths(
        self,
        num_prompts: int,
        input_len: int,
        output_len: int,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            build_serving_requests(
                dataset_name="random",
                num_prompts=num_prompts,
                input_len=input_len,
                output_len=output_len,
                model="demo-model",
                stream=True,
            )


class TestYearDemoWorkloads:
    """Year Demo Workloads 测试。"""

    def test_year1_workloads(self) -> None:
        """测试 Year1 workloads 获取。"""
        workloads = get_year1_workloads()

        assert len(workloads) == 3

        # 验证类型
        types = {w.workload_type for w in workloads}
        assert types == {WorkloadType.SHORT, WorkloadType.LONG, WorkloadType.STRESS}

        # 验证 SHORT
        short = next(w for w in workloads if w.workload_type == WorkloadType.SHORT)
        assert short.prompt_len == 128
        assert short.output_len == 128

        # 验证 LONG
        long_w = next(w for w in workloads if w.workload_type == WorkloadType.LONG)
        assert long_w.prompt_len == 2048
        assert long_w.output_len == 512

        # 验证 STRESS
        stress = next(w for w in workloads if w.workload_type == WorkloadType.STRESS)
        assert stress.concurrent is True
        assert stress.kv_budget_tokens == 4096

    def test_year2_workloads(self) -> None:
        """测试 Year2 workloads。"""
        workloads = get_year2_workloads()
        assert len(workloads) == 3

        long_w = next(w for w in workloads if w.workload_type == WorkloadType.LONG)
        assert long_w.prompt_len == 8192  # 8K context

    def test_year3_workloads(self) -> None:
        """测试 Year3 workloads。"""
        workloads = get_year3_workloads()
        assert len(workloads) == 3

        long_w = next(w for w in workloads if w.workload_type == WorkloadType.LONG)
        assert long_w.prompt_len == 32768  # 32K context

    def test_get_workloads_by_year(self) -> None:
        """测试按年份获取 workloads。"""
        for year in [1, 2, 3]:
            workloads = get_workloads_by_year(year)
            assert len(workloads) == 3

        with pytest.raises(ValueError, match="Invalid year"):
            get_workloads_by_year(4)

    def test_get_workload_by_type(self) -> None:
        """测试按类型获取 workload。"""
        spec = get_workload_by_type(1, WorkloadType.SHORT)
        assert spec.name == "year1_short"
        assert spec.workload_type == WorkloadType.SHORT

        with pytest.raises(ValueError):
            get_workload_by_type(99, WorkloadType.SHORT)

    def test_year1_workloads_constant(self) -> None:
        """测试 YEAR1_WORKLOADS 常量。"""
        assert len(YEAR1_WORKLOADS) == 3


class TestCreateCustomWorkload:
    """自定义 Workload 创建测试。"""

    def test_create_valid(self) -> None:
        """测试创建有效 workload。"""
        spec = create_custom_workload(
            name="custom",
            workload_type=WorkloadType.SHORT,
            prompt_len=200,
            output_len=100,
            num_requests=10,
            concurrent=True,
            kv_budget_tokens=1024,
        )

        assert spec.name == "custom"
        assert spec.prompt_len == 200
        assert spec.concurrent is True
        assert spec.kv_budget_tokens == 1024

    def test_create_invalid_prompt_len(self) -> None:
        """测试无效 prompt_len。"""
        with pytest.raises(ValueError, match="prompt_len must be positive"):
            create_custom_workload(
                name="invalid",
                workload_type=WorkloadType.SHORT,
                prompt_len=-10,
                output_len=50,
                num_requests=1,
            )

    def test_create_invalid_kv_budget(self) -> None:
        """测试无效 kv_budget_tokens。"""
        with pytest.raises(ValueError, match="kv_budget_tokens must be positive"):
            create_custom_workload(
                name="invalid",
                workload_type=WorkloadType.SHORT,
                prompt_len=100,
                output_len=50,
                num_requests=1,
                kv_budget_tokens=-1,
            )


class TestIntegration:
    """集成测试 - 端到端采样流程。"""

    def test_year1_demo_sampling(self) -> None:
        """测试 Year1 Demo 完整采样流程。

        这是任务书验收用例的实现。
        """
        # 构造 Year1 workload
        workloads = get_year1_workloads()
        assert len(workloads) == 3

        # 使用 RandomDataset 采样
        dataset = RandomDataset(seed=42)

        for spec in workloads:
            # 采样 5 条 request
            requests = dataset.sample(
                WorkloadSpec(
                    name=spec.name,
                    workload_type=spec.workload_type,
                    prompt_len=spec.prompt_len,
                    output_len=spec.output_len,
                    num_requests=5,
                    concurrent=spec.concurrent,
                    kv_budget_tokens=spec.kv_budget_tokens,
                )
            )

            assert len(requests) == 5

            # 打印 request_id 与 prompt 长度（验收输出）
            print(f"\n{spec.name}:")
            for req in requests:
                print(f"  - {req.request_id}: prompt_len={len(req.prompt)}")

            # 验证每个 request
            for req in requests:
                assert isinstance(req, BenchmarkRequest)
                uuid.UUID(req.request_id)  # 有效 UUID
                assert len(req.prompt) > 0
                assert req.max_tokens == spec.output_len

    def test_benchmark_dataset_interface(self) -> None:
        """测试 BenchmarkDataset 接口一致性。"""
        datasets: list[BenchmarkDataset] = [
            RandomDataset(seed=42),
            SyntheticShareGPTDataset(seed=42),
        ]

        spec = WorkloadSpec(
            name="interface_test",
            workload_type=WorkloadType.SHORT,
            prompt_len=100,
            output_len=50,
            num_requests=3,
        )

        for dataset in datasets:
            # 验证接口
            assert hasattr(dataset, "name")
            assert hasattr(dataset, "sample")

            # 验证返回值
            requests = dataset.sample(spec)
            assert isinstance(requests, list)
            assert len(requests) == 3
            assert all(isinstance(r, BenchmarkRequest) for r in requests)
