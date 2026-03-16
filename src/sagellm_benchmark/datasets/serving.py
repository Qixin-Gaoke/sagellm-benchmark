"""Serving benchmark dataset helpers.

复用现有 random/sharegpt 数据集实现，为 serving benchmark 主线生成请求。
"""

from __future__ import annotations

from pathlib import Path

from sagellm_benchmark.datasets.base import BenchmarkDataset
from sagellm_benchmark.datasets.random import RandomDataset
from sagellm_benchmark.datasets.sharegpt import ShareGPTDataset
from sagellm_benchmark.types import BenchmarkRequest, WorkloadSpec, WorkloadType

DEFAULT_SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DEFAULT_SHAREGPT_SPLIT = "train[:1000]"
DEFAULT_SHAREGPT_MIN_PROMPT_LEN = 50
DEFAULT_SHAREGPT_MAX_PROMPT_LEN = 5000
SERVING_DATASET_NAMES = ("random", "sharegpt")


def load_serving_dataset(
    dataset_name: str,
    *,
    seed: int | None = 0,
    sharegpt_path: str | Path | None = None,
) -> BenchmarkDataset:
    """加载 serving benchmark 使用的数据集."""
    if dataset_name == "random":
        return RandomDataset(seed=seed)

    if dataset_name == "sharegpt":
        if sharegpt_path is not None:
            return ShareGPTDataset.from_file(
                sharegpt_path,
                seed=seed,
                min_prompt_len=DEFAULT_SHAREGPT_MIN_PROMPT_LEN,
                max_prompt_len=DEFAULT_SHAREGPT_MAX_PROMPT_LEN,
            )
        return ShareGPTDataset.from_huggingface(
            repo_id=DEFAULT_SHAREGPT_REPO_ID,
            split=DEFAULT_SHAREGPT_SPLIT,
            seed=seed,
            min_prompt_len=DEFAULT_SHAREGPT_MIN_PROMPT_LEN,
            max_prompt_len=DEFAULT_SHAREGPT_MAX_PROMPT_LEN,
        )

    supported = ", ".join(SERVING_DATASET_NAMES)
    raise ValueError(f"Unsupported serving dataset '{dataset_name}'. Expected one of: {supported}")


def build_serving_requests(
    *,
    dataset_name: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    model: str,
    stream: bool,
    seed: int | None = 0,
    sharegpt_path: str | Path | None = None,
) -> list[BenchmarkRequest]:
    """从现有数据集能力生成 serving benchmark 请求列表."""
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, got {num_prompts}")
    if input_len <= 0:
        raise ValueError(f"input_len must be positive, got {input_len}")
    if output_len <= 0:
        raise ValueError(f"output_len must be positive, got {output_len}")

    dataset = load_serving_dataset(
        dataset_name,
        seed=seed,
        sharegpt_path=sharegpt_path,
    )
    spec = WorkloadSpec(
        name=f"serving_{dataset_name}",
        workload_type=WorkloadType.SHORT,
        prompt_len=input_len,
        output_len=output_len,
        num_requests=num_prompts,
    )
    requests = dataset.sample(spec)

    return [
        BenchmarkRequest(
            prompt=request.prompt,
            max_tokens=output_len,
            request_id=f"{dataset_name}-{index}",
            model=model,
            stream=stream,
            temperature=request.temperature,
            top_p=request.top_p,
            kv_budget_tokens=request.kv_budget_tokens,
        )
        for index, request in enumerate(requests)
    ]
