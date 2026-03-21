"""Tests for profile-first workload planning."""

from __future__ import annotations

import pytest

from sagellm_benchmark.workload_profiles import (
    MAINLINE_SCENARIO_SOURCE,
    SUPPLEMENT_SCENARIO_SOURCE,
    build_execution_plan,
)


def test_profile_default_vllm_random_plan() -> None:
    plan = build_execution_plan(
        profile_id="vllm_random",
        supplements=(),
        dataset_path=None,
        num_prompts=None,
        input_len=None,
        output_len=None,
        batch_sizes=(1, 2, 4),
        mode="live-compare",
    )

    assert plan.profile.profile_id == "vllm_random"
    assert plan.profile.dataset_name == "random"
    assert plan.profile.scenario_source == MAINLINE_SCENARIO_SOURCE
    assert len(plan.scenarios) == 3
    assert all(item.scenario_source == MAINLINE_SCENARIO_SOURCE for item in plan.scenarios)


def test_profile_custom_fail_fast_requires_all_fields() -> None:
    with pytest.raises(ValueError, match="vllm_custom requires --dataset-path"):
        build_execution_plan(
            profile_id="vllm_custom",
            supplements=(),
            dataset_path=None,
            num_prompts=8,
            input_len=128,
            output_len=128,
            batch_sizes=(1,),
            mode="live-compare",
        )


def test_profile_unknown_fails_fast() -> None:
    with pytest.raises(ValueError, match="unknown profile_id"):
        build_execution_plan(
            profile_id="short",
            supplements=(),
            dataset_path=None,
            num_prompts=None,
            input_len=None,
            output_len=None,
            batch_sizes=(1,),
            mode="live-compare",
        )


def test_q1q8_supplement_injection() -> None:
    plan = build_execution_plan(
        profile_id="vllm_random",
        supplements=("q1q8_supplement",),
        dataset_path=None,
        num_prompts=None,
        input_len=None,
        output_len=None,
        batch_sizes=(1,),
        mode="live-compare",
    )

    # 1 mainline + 8 supplement scenarios at batch size 1
    assert len(plan.scenarios) == 9
    supplement_rows = [
        row for row in plan.scenarios if row.scenario_source == SUPPLEMENT_SCENARIO_SOURCE
    ]
    assert len(supplement_rows) == 8
    assert all(row.workload_profile == "vllm_random" for row in supplement_rows)
