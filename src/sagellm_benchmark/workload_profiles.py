"""Profile-first workload planning for compare/run pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAINLINE_SCENARIO_SOURCE = "mainline"
SUPPLEMENT_SCENARIO_SOURCE = "supplement"
Q1Q8_SUPPLEMENT_ID = "q1q8_supplement"


@dataclass(frozen=True)
class WorkloadProfile:
    """Unified benchmark workload profile."""

    profile_id: str
    dataset_name: str
    dataset_path: str
    num_prompts: int
    input_len: int
    output_len: int
    batch_sizes: tuple[int, ...]
    mode: str
    scenario_source: str = MAINLINE_SCENARIO_SOURCE


@dataclass(frozen=True)
class WorkloadScenarioPlan:
    """One executable scenario generated from profile + supplements."""

    scenario_name: str
    scenario_source: str
    dataset_name: str
    dataset_path: str
    num_prompts: int
    input_len: int
    output_len: int
    batch_size: int
    workload_profile: str
    supplements: tuple[str, ...]


@dataclass(frozen=True)
class WorkloadExecutionPlan:
    """Resolved execution plan shared by compare and run."""

    profile: WorkloadProfile
    supplements: tuple[str, ...]
    scenarios: tuple[WorkloadScenarioPlan, ...]


_Q1Q8_DEFS: tuple[dict[str, Any], ...] = (
    {"name": "Q1", "input_len": 32, "output_len": 64, "num_prompts": 5},
    {"name": "Q2", "input_len": 512, "output_len": 128, "num_prompts": 3},
    {"name": "Q3", "input_len": 128, "output_len": 256, "num_prompts": 3},
    {"name": "Q4", "input_len": 256, "output_len": 256, "num_prompts": 3},
    {"name": "Q5", "input_len": 32, "output_len": 64, "num_prompts": 10},
    {"name": "Q6", "input_len": 512, "output_len": 256, "num_prompts": 10},
    {"name": "Q7", "input_len": 256, "output_len": 512, "num_prompts": 3},
    {"name": "Q8", "input_len": 192, "output_len": 128, "num_prompts": 4},
)


def resolve_profile(
    *,
    profile_id: str,
    dataset_path: str | None,
    num_prompts: int | None,
    input_len: int | None,
    output_len: int | None,
    batch_sizes: tuple[int, ...] | None,
    mode: str,
) -> WorkloadProfile:
    """Resolve one profile with strict fail-fast validation."""

    normalized_profile = profile_id.strip()
    if not normalized_profile:
        raise ValueError("profile_id is required")

    if mode.strip() not in {"live", "traffic", "batch", "live-compare"}:
        raise ValueError(f"unsupported mode: {mode}")

    defaults: dict[str, WorkloadProfile] = {
        "vllm_random": WorkloadProfile(
            profile_id="vllm_random",
            dataset_name="random",
            dataset_path="builtin://random",
            num_prompts=64,
            input_len=128,
            output_len=128,
            batch_sizes=(1, 2, 4),
            mode=mode,
        ),
        "vllm_sharegpt": WorkloadProfile(
            profile_id="vllm_sharegpt",
            dataset_name="sharegpt",
            dataset_path="hf://anon8231489123/ShareGPT_Vicuna_unfiltered:train[:1000]",
            num_prompts=64,
            input_len=256,
            output_len=128,
            batch_sizes=(1, 2, 4),
            mode=mode,
        ),
        "vllm_hf": WorkloadProfile(
            profile_id="vllm_hf",
            dataset_name="hf",
            dataset_path="hf://anon8231489123/ShareGPT_Vicuna_unfiltered:train[:1000]",
            num_prompts=64,
            input_len=256,
            output_len=128,
            batch_sizes=(1, 2, 4),
            mode=mode,
        ),
    }

    if normalized_profile in defaults:
        base = defaults[normalized_profile]
        return WorkloadProfile(
            profile_id=base.profile_id,
            dataset_name=base.dataset_name,
            dataset_path=base.dataset_path,
            num_prompts=num_prompts if num_prompts is not None else base.num_prompts,
            input_len=input_len if input_len is not None else base.input_len,
            output_len=output_len if output_len is not None else base.output_len,
            batch_sizes=batch_sizes if batch_sizes else base.batch_sizes,
            mode=mode,
            scenario_source=MAINLINE_SCENARIO_SOURCE,
        )

    if normalized_profile != "vllm_custom":
        raise ValueError(
            "unknown profile_id; supported profiles: "
            "vllm_random, vllm_sharegpt, vllm_hf, vllm_custom"
        )

    if dataset_path is None or not dataset_path.strip():
        raise ValueError("vllm_custom requires --dataset-path")
    if num_prompts is None:
        raise ValueError("vllm_custom requires --num-prompts")
    if input_len is None:
        raise ValueError("vllm_custom requires --input-len")
    if output_len is None:
        raise ValueError("vllm_custom requires --output-len")

    resolved_batch_sizes = batch_sizes if batch_sizes else (1, 2, 4)

    return WorkloadProfile(
        profile_id="vllm_custom",
        dataset_name="custom",
        dataset_path=dataset_path.strip(),
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        batch_sizes=resolved_batch_sizes,
        mode=mode,
        scenario_source=MAINLINE_SCENARIO_SOURCE,
    )


def normalize_supplements(raw_supplements: tuple[str, ...] | None) -> tuple[str, ...]:
    if not raw_supplements:
        return ()

    normalized = tuple(item.strip() for item in raw_supplements if item and item.strip())
    unsupported = [item for item in normalized if item != Q1Q8_SUPPLEMENT_ID]
    if unsupported:
        raise ValueError(
            "unsupported supplement id(s): "
            + ", ".join(sorted(set(unsupported)))
            + "; supported: q1q8_supplement"
        )
    return tuple(dict.fromkeys(normalized))


def build_execution_plan(
    *,
    profile_id: str,
    supplements: tuple[str, ...] | None,
    dataset_path: str | None,
    num_prompts: int | None,
    input_len: int | None,
    output_len: int | None,
    batch_sizes: tuple[int, ...] | None,
    mode: str,
) -> WorkloadExecutionPlan:
    """Build one strict, traceable execution plan."""

    profile = resolve_profile(
        profile_id=profile_id,
        dataset_path=dataset_path,
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        batch_sizes=batch_sizes,
        mode=mode,
    )
    resolved_supplements = normalize_supplements(supplements)

    scenarios: list[WorkloadScenarioPlan] = []
    for batch_size in profile.batch_sizes:
        scenarios.append(
            WorkloadScenarioPlan(
                scenario_name=f"{profile.profile_id}_b{batch_size}",
                scenario_source=MAINLINE_SCENARIO_SOURCE,
                dataset_name=profile.dataset_name,
                dataset_path=profile.dataset_path,
                num_prompts=profile.num_prompts,
                input_len=profile.input_len,
                output_len=profile.output_len,
                batch_size=batch_size,
                workload_profile=profile.profile_id,
                supplements=resolved_supplements,
            )
        )

    if Q1Q8_SUPPLEMENT_ID in resolved_supplements:
        for batch_size in profile.batch_sizes:
            for query in _Q1Q8_DEFS:
                scenarios.append(
                    WorkloadScenarioPlan(
                        scenario_name=f"{query['name'].lower()}_b{batch_size}",
                        scenario_source=SUPPLEMENT_SCENARIO_SOURCE,
                        dataset_name=profile.dataset_name,
                        dataset_path=profile.dataset_path,
                        num_prompts=int(query["num_prompts"]),
                        input_len=int(query["input_len"]),
                        output_len=int(query["output_len"]),
                        batch_size=batch_size,
                        workload_profile=profile.profile_id,
                        supplements=resolved_supplements,
                    )
                )

    return WorkloadExecutionPlan(
        profile=profile,
        supplements=resolved_supplements,
        scenarios=tuple(scenarios),
    )


def profile_to_serving_dataset(profile: WorkloadProfile) -> tuple[str, str | None]:
    """Map profile dataset to serving dataset loader inputs."""

    if profile.dataset_name == "random":
        return "random", None
    if profile.dataset_name in {"sharegpt", "hf"}:
        return "sharegpt", None
    if profile.dataset_name == "custom":
        return "sharegpt", profile.dataset_path
    raise ValueError(f"unsupported profile dataset_name: {profile.dataset_name}")
