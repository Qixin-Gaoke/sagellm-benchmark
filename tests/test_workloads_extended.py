"""Additional profile-first workload planning tests."""

from __future__ import annotations

import pytest

from sagellm_benchmark.workload_profiles import build_execution_plan


def test_run_compare_plan_consistency() -> None:
    run_plan = build_execution_plan(
        profile_id="vllm_sharegpt",
        supplements=("q1q8_supplement",),
        dataset_path=None,
        num_prompts=16,
        input_len=192,
        output_len=96,
        batch_sizes=(1, 2),
        mode="traffic",
    )
    compare_plan = build_execution_plan(
        profile_id="vllm_sharegpt",
        supplements=("q1q8_supplement",),
        dataset_path=None,
        num_prompts=16,
        input_len=192,
        output_len=96,
        batch_sizes=(1, 2),
        mode="live-compare",
    )

    assert run_plan.profile.profile_id == compare_plan.profile.profile_id
    assert [row.scenario_name for row in run_plan.scenarios] == [
        row.scenario_name for row in compare_plan.scenarios
    ]


def test_unsupported_supplement_fails_fast() -> None:
    with pytest.raises(ValueError, match="unsupported supplement id"):
        build_execution_plan(
            profile_id="vllm_random",
            supplements=("q9",),
            dataset_path=None,
            num_prompts=None,
            input_len=None,
            output_len=None,
            batch_sizes=(1,),
            mode="live-compare",
        )
