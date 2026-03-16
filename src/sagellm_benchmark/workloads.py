"""Benchmark workloads and configurations.

Supports predefined workloads, YAML/JSON config loading, and template generation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WorkloadType(StrEnum):
    """Workload types for benchmarking."""

    QUERY = "query"  # TPCH/TPCC-style query workload
    SHORT = "short"  # Short prompt + short output
    LONG = "long"  # Long prompt + medium output
    STRESS = "stress"  # Concurrent requests, pressure test
    STREAMING = "streaming"  # Streaming / SSE output test
    BATCH_INFERENCE = "batch_inference"  # Offline batch inference throughput
    MIXED = "mixed"  # Mixed request types


class WorkloadQuery(StrEnum):
    """Query-style workload identifiers."""

    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    Q5 = "Q5"
    Q6 = "Q6"
    Q7 = "Q7"
    Q8 = "Q8"


@dataclass
class WorkloadConfig:
    """Configuration for a benchmark workload.

    Attributes:
        name: Workload identifier.
        workload_type: Type of workload (short/long/stress/streaming/batch_inference/mixed).
        prompt: Input prompt text.
        prompt_tokens: Expected prompt token count (approximate).
        max_tokens: Maximum tokens to generate.
        num_requests: Number of requests to run.
        concurrent: Whether to run requests concurrently.
        temperature: Sampling temperature (0.0 = greedy).
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter (None = disabled).
        repetition_penalty: Repetition penalty factor (1.0 = no penalty).
        stream: Whether to request streaming (SSE) output.
        warmup_rounds: Number of warmup rounds before measurement.
        concurrency: Explicit concurrency level (overrides ``concurrent`` flag).
        extra_params: Additional backend-specific parameters.
    """

    name: str
    workload_type: WorkloadType
    prompt: str
    prompt_tokens: int
    max_tokens: int
    num_requests: int = 1
    concurrent: bool = False
    temperature: float | None = None  # None = use model default (greedy)
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    stream: bool = False
    warmup_rounds: int = 0
    concurrency: int | None = None  # explicit concurrency level

    # Additional params
    extra_params: dict[str, Any] = field(default_factory=dict)


# Legacy workloads (deprecated – retained only for backward compatibility)
# Use TPCH_WORKLOADS (Q1-Q8) for all new benchmarks.
_LEGACY_WORKLOADS = [
    WorkloadConfig(
        name="short_input",
        workload_type=WorkloadType.SHORT,
        prompt="Hello world, tell me a short story.",
        prompt_tokens=128,
        max_tokens=128,
        num_requests=5,
    ),
    WorkloadConfig(
        name="long_input",
        workload_type=WorkloadType.LONG,
        prompt=" ".join(["This is context about AI and technology."] * 20),
        prompt_tokens=200,
        max_tokens=200,
        num_requests=3,
    ),
    WorkloadConfig(
        name="stress_test",
        workload_type=WorkloadType.STRESS,
        prompt="Write a poem about AI.",
        prompt_tokens=256,
        max_tokens=256,
        num_requests=10,
        concurrent=True,
    ),
]

# Backward-compatible aliases (deprecated)
YEAR1_WORKLOADS = _LEGACY_WORKLOADS
M1_WORKLOADS = _LEGACY_WORKLOADS


# TPCH/TPCC-style query workloads
TPCH_WORKLOADS = [
    WorkloadConfig(
        name=WorkloadQuery.Q1.value,
        workload_type=WorkloadType.QUERY,
        prompt="用一句话回答：什么是 Transformer？",
        prompt_tokens=32,
        max_tokens=64,
        num_requests=5,
    ),
    WorkloadConfig(
        name=WorkloadQuery.Q2.value,
        workload_type=WorkloadType.QUERY,
        prompt="\n".join(
            [
                "请阅读以下长上下文并做摘要：",
                " ".join(
                    [
                        "大型语言模型在推理系统中需要考虑吞吐、延迟、显存占用和可扩展性。"
                        "调度器需要平衡 prefilling 和 decoding，避免 head-of-line blocking。"
                    ]
                    * 12
                ),
            ]
        ),
        prompt_tokens=512,
        max_tokens=128,
        num_requests=3,
    ),
    WorkloadConfig(
        name=WorkloadQuery.Q3.value,
        workload_type=WorkloadType.QUERY,
        prompt="写一个 Python 函数，输入整数数组，返回前缀和数组，并给出时间复杂度。",
        prompt_tokens=128,
        max_tokens=256,
        num_requests=3,
    ),
    WorkloadConfig(
        name=WorkloadQuery.Q4.value,
        workload_type=WorkloadType.QUERY,
        prompt=(
            "你是一个技术助手。\n"
            "用户: 我在做 LLM 推理性能优化。\n"
            "助手: 你更关注延迟还是吞吐？\n"
            "用户: 两者都要兼顾，请给我一个分步骤方案。"
        ),
        prompt_tokens=256,
        max_tokens=256,
        num_requests=3,
    ),
    WorkloadConfig(
        name=WorkloadQuery.Q5.value,
        workload_type=WorkloadType.QUERY,
        prompt="请给我 3 条提升 API 稳定性的建议。",
        prompt_tokens=32,
        max_tokens=64,
        num_requests=10,
        concurrent=True,
    ),
    WorkloadConfig(
        name=WorkloadQuery.Q6.value,
        workload_type=WorkloadType.QUERY,
        prompt=" ".join(["分析分布式推理系统中的瓶颈与优化策略。"] * 24),
        prompt_tokens=512,
        max_tokens=256,
        num_requests=10,
        concurrent=True,
    ),
    WorkloadConfig(
        name=WorkloadQuery.Q7.value,
        workload_type=WorkloadType.QUERY,
        prompt="请逐步推理：比较同步和异步批处理在高并发场景下的优缺点。",
        prompt_tokens=256,
        max_tokens=512,
        num_requests=3,
    ),
    WorkloadConfig(
        name=WorkloadQuery.Q8.value,
        workload_type=WorkloadType.QUERY,
        prompt="综合任务：总结、分类并给出执行建议（兼顾准确性和时延）。",
        prompt_tokens=192,
        max_tokens=128,
        num_requests=4,
        concurrent=True,
    ),
]


def get_workloads_by_selector(selector: str) -> list[WorkloadConfig]:
    """Resolve workload selector to workload config list.

    Args:
        selector: Workload selector string from CLI.

    Returns:
        List of workload configs to run.
    """
    selected = selector.lower()

    if selected in {"all", "query"}:
        return TPCH_WORKLOADS
    # Legacy selectors (deprecated – prefer Q1-Q8 or 'all')
    if selected in {"m1", "year1"}:
        import warnings

        warnings.warn(
            "'year1'/'m1' workloads are deprecated. Use '--workload all' for Q1-Q8.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _LEGACY_WORKLOADS
    if selected in {"short", "long", "stress"}:
        import warnings

        warnings.warn(
            f"'{selected}' workload is deprecated. Use Q1-Q8 workloads instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [
            workload for workload in _LEGACY_WORKLOADS if workload.workload_type.value == selected
        ]
    if selected == "streaming":
        return STREAMING_WORKLOADS
    if selected in {"batch", "batch_inference"}:
        return BATCH_INFERENCE_WORKLOADS
    if selected == "mixed":
        return MIXED_WORKLOADS

    for workload in TPCH_WORKLOADS:
        if workload.name.lower() == selected:
            return [workload]

    raise ValueError(f"Unknown workload selector: {selector}")


# ---------------------------------------------------------------------------
# Streaming workloads (SSE / token-by-token)
# ---------------------------------------------------------------------------

STREAMING_WORKLOADS = [
    WorkloadConfig(
        name="streaming_short",
        workload_type=WorkloadType.STREAMING,
        prompt="Tell me a short story about AI.",
        prompt_tokens=32,
        max_tokens=128,
        num_requests=5,
        stream=True,
        warmup_rounds=1,
    ),
    WorkloadConfig(
        name="streaming_long",
        workload_type=WorkloadType.STREAMING,
        prompt=" ".join(["Explain the principles of distributed AI inference systems."] * 5),
        prompt_tokens=256,
        max_tokens=256,
        num_requests=3,
        stream=True,
        warmup_rounds=1,
    ),
    WorkloadConfig(
        name="streaming_concurrent",
        workload_type=WorkloadType.STREAMING,
        prompt="Summarize the key challenges in LLM serving.",
        prompt_tokens=64,
        max_tokens=128,
        num_requests=10,
        concurrent=True,
        concurrency=4,
        stream=True,
        warmup_rounds=2,
    ),
]

# ---------------------------------------------------------------------------
# Batch inference workloads (local batch submission)
# ---------------------------------------------------------------------------

BATCH_INFERENCE_WORKLOADS = [
    WorkloadConfig(
        name="batch_small",
        workload_type=WorkloadType.BATCH_INFERENCE,
        prompt="What is the capital of France?",
        prompt_tokens=16,
        max_tokens=32,
        num_requests=20,
        concurrent=True,
        concurrency=8,
        warmup_rounds=1,
    ),
    WorkloadConfig(
        name="batch_medium",
        workload_type=WorkloadType.BATCH_INFERENCE,
        prompt=" ".join(["Analyze the performance characteristics of large language models."] * 4),
        prompt_tokens=128,
        max_tokens=128,
        num_requests=16,
        concurrent=True,
        concurrency=8,
        warmup_rounds=1,
    ),
    WorkloadConfig(
        name="batch_large",
        workload_type=WorkloadType.BATCH_INFERENCE,
        prompt=" ".join(["This is a long context prompt for batch inference testing."] * 15),
        prompt_tokens=512,
        max_tokens=256,
        num_requests=8,
        concurrent=True,
        concurrency=4,
        warmup_rounds=1,
    ),
]

# ---------------------------------------------------------------------------
# Mixed workloads (heterogeneous request types)
# ---------------------------------------------------------------------------

MIXED_WORKLOADS = [
    WorkloadConfig(
        name="mixed_short_stream",
        workload_type=WorkloadType.MIXED,
        prompt="Give me a brief overview of transformer architecture.",
        prompt_tokens=32,
        max_tokens=64,
        num_requests=10,
        concurrent=True,
        concurrency=2,
        stream=True,
        temperature=0.7,
        top_k=50,
        warmup_rounds=1,
    ),
    WorkloadConfig(
        name="mixed_long_batch",
        workload_type=WorkloadType.MIXED,
        prompt=" ".join(
            ["Discuss the trade-offs between latency and throughput in AI inference."] * 6
        ),
        prompt_tokens=256,
        max_tokens=256,
        num_requests=6,
        concurrent=True,
        concurrency=3,
        temperature=0.9,
        repetition_penalty=1.1,
        warmup_rounds=1,
    ),
    WorkloadConfig(
        name="mixed_greedy_sampling",
        workload_type=WorkloadType.MIXED,
        prompt="List 5 key optimizations for LLM serving systems.",
        prompt_tokens=48,
        max_tokens=128,
        num_requests=8,
        concurrent=True,
        concurrency=4,
        temperature=None,  # greedy
        top_p=1.0,
        top_k=None,
        warmup_rounds=1,
    ),
]


# ---------------------------------------------------------------------------
# WorkloadLoader — load from YAML / JSON file
# ---------------------------------------------------------------------------


class WorkloadLoader:
    """Load workload configurations from YAML or JSON files.

    Example YAML file::

        workloads:
          - name: custom_short
            workload_type: short
            prompt: "Hello, how are you?"
            prompt_tokens: 16
            max_tokens: 64
            num_requests: 5
            temperature: 0.7
            top_k: 40
            warmup_rounds: 1

    Example JSON file::

        {
          "workloads": [
            {
              "name": "custom_short",
              "workload_type": "short",
              "prompt": "Hello, how are you?",
              "prompt_tokens": 16,
              "max_tokens": 64,
              "num_requests": 5
            }
          ]
        }
    """

    @classmethod
    def load(cls, path: str | Path) -> list[WorkloadConfig]:
        """Load workload configs from a YAML or JSON file.

        Args:
            path: Path to YAML or JSON file.

        Returns:
            List of WorkloadConfig objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported or data is invalid.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Workload config file not found: {p}")

        suffix = p.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return cls._load_yaml(p)
        elif suffix == ".json":
            return cls._load_json(p)
        else:
            raise ValueError(
                f"Unsupported workload config format: {suffix} (expected .yaml/.yml/.json)"
            )

    @classmethod
    def _load_yaml(cls, path: Path) -> list[WorkloadConfig]:
        try:
            import yaml  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML workload configs. Install with: pip install pyyaml"
            ) from e
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls._parse_data(data)

    @classmethod
    def _load_json(cls, path: Path) -> list[WorkloadConfig]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls._parse_data(data)

    @classmethod
    def _parse_data(cls, data: Any) -> list[WorkloadConfig]:
        """Parse raw dict/list data into WorkloadConfig objects."""
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("workloads", [])
        else:
            raise ValueError(f"Expected list or dict with 'workloads' key, got {type(data)}")

        configs = []
        for entry in entries:
            wt_str = entry.get("workload_type", "short")
            try:
                wt = WorkloadType(wt_str.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown workload_type '{wt_str}'. Valid values: "
                    + ", ".join(str(v) for v in WorkloadType)
                )

            cfg = WorkloadConfig(
                name=entry["name"],
                workload_type=wt,
                prompt=entry["prompt"],
                prompt_tokens=int(entry.get("prompt_tokens", 64)),
                max_tokens=int(entry.get("max_tokens", 128)),
                num_requests=int(entry.get("num_requests", 1)),
                concurrent=bool(entry.get("concurrent", False)),
                temperature=entry.get("temperature", None),
                top_p=float(entry.get("top_p", 1.0)),
                top_k=entry.get("top_k", None),
                repetition_penalty=float(entry.get("repetition_penalty", 1.0)),
                stream=bool(entry.get("stream", False)),
                warmup_rounds=int(entry.get("warmup_rounds", 0)),
                concurrency=entry.get("concurrency", None),
                extra_params=entry.get("extra_params", {}),
            )
            configs.append(cfg)
            logger.debug(f"Loaded workload: {cfg.name} ({cfg.workload_type})")

        return configs


# ---------------------------------------------------------------------------
# WorkloadTemplateGenerator — generate starter templates
# ---------------------------------------------------------------------------


class WorkloadTemplateGenerator:
    """Generate workload template files to help users get started.

    Usage::

        WorkloadTemplateGenerator.generate_yaml("/path/to/my_workloads.yaml")
        WorkloadTemplateGenerator.generate_json("/path/to/my_workloads.json")
    """

    _TEMPLATE_WORKLOADS: list[WorkloadConfig] = [
        WorkloadConfig(
            name="custom_short",
            workload_type=WorkloadType.SHORT,
            prompt="Hello, tell me about AI.",
            prompt_tokens=16,
            max_tokens=64,
            num_requests=5,
            temperature=0.7,
            top_k=40,
            warmup_rounds=1,
        ),
        WorkloadConfig(
            name="custom_streaming",
            workload_type=WorkloadType.STREAMING,
            prompt="Explain transformer architecture in detail.",
            prompt_tokens=32,
            max_tokens=256,
            num_requests=3,
            stream=True,
            warmup_rounds=1,
        ),
        WorkloadConfig(
            name="custom_batch",
            workload_type=WorkloadType.BATCH_INFERENCE,
            prompt="Summarize the key ideas in deep learning.",
            prompt_tokens=32,
            max_tokens=128,
            num_requests=16,
            concurrent=True,
            concurrency=8,
            warmup_rounds=2,
        ),
    ]

    @classmethod
    def generate_json(cls, output_path: str | Path) -> str:
        """Generate a JSON template file.

        Args:
            output_path: Path to write the template to.

        Returns:
            The JSON string written to the file.
        """
        data = {"workloads": [asdict(w) for w in cls._TEMPLATE_WORKLOADS]}
        content = json.dumps(data, indent=2, ensure_ascii=False)
        Path(output_path).write_text(content, encoding="utf-8")
        logger.info(f"Workload template written to {output_path}")
        return content

    @classmethod
    def generate_yaml(cls, output_path: str | Path) -> str:
        """Generate a YAML template file.

        Args:
            output_path: Path to write the template to.

        Returns:
            The YAML string written to the file.
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError as e:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

        data = {"workloads": [asdict(w) for w in cls._TEMPLATE_WORKLOADS]}
        content = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
        Path(output_path).write_text(content, encoding="utf-8")
        logger.info(f"Workload template written to {output_path}")
        return content
