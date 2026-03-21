"""Datasets 模块 - 提供数据集加载与采样功能。

此模块提供：
- BenchmarkDataset: 数据集抽象基类
- RandomDataset: 随机数据生成
- ShareGPTDataset: ShareGPT 真实数据加载
- SyntheticShareGPTDataset: 合成 ShareGPT 数据
- Year Demo Workloads: 预置的 Year1/2/3 Workload 规格

Example:
    >>> from sagellm_benchmark.datasets import RandomDataset, get_year1_workloads
    >>> dataset = RandomDataset(seed=42)
    >>> workloads = get_year1_workloads()
    >>> requests = dataset.sample(workloads[0])
"""

from __future__ import annotations

from sagellm_benchmark.datasets.base import BenchmarkDataset
from sagellm_benchmark.datasets.random import RandomDataset
from sagellm_benchmark.datasets.serving import (
    SERVING_DATASET_NAMES,
    build_serving_requests,
    load_serving_dataset,
)
from sagellm_benchmark.datasets.sharegpt import ShareGPTDataset, SyntheticShareGPTDataset
from sagellm_benchmark.datasets.year_demo import (
    YEAR1_LONG,
    YEAR1_SHORT,
    YEAR1_STRESS,
    YEAR1_WORKLOADS,
    YEAR2_LONG,
    YEAR2_SHORT,
    YEAR2_STRESS,
    YEAR2_WORKLOADS,
    YEAR3_LONG,
    YEAR3_SHORT,
    YEAR3_STRESS,
    YEAR3_WORKLOADS,
    create_custom_workload,
    get_workload_by_type,
    get_workloads_by_year,
    get_year1_workloads,
    get_year2_workloads,
    get_year3_workloads,
)

__all__ = [
    # 抽象基类
    "BenchmarkDataset",
    # 数据集实现
    "RandomDataset",
    "ShareGPTDataset",
    "SyntheticShareGPTDataset",
    "SERVING_DATASET_NAMES",
    "load_serving_dataset",
    "build_serving_requests",
    # Year Demo Workloads
    "YEAR1_WORKLOADS",
    "YEAR1_SHORT",
    "YEAR1_LONG",
    "YEAR1_STRESS",
    "YEAR2_WORKLOADS",
    "YEAR2_SHORT",
    "YEAR2_LONG",
    "YEAR2_STRESS",
    "YEAR3_WORKLOADS",
    "YEAR3_SHORT",
    "YEAR3_LONG",
    "YEAR3_STRESS",
    # 辅助函数
    "get_year1_workloads",
    "get_year2_workloads",
    "get_year3_workloads",
    "get_workloads_by_year",
    "get_workload_by_type",
    "create_custom_workload",
]
