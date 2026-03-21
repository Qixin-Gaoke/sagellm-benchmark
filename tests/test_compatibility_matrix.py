from __future__ import annotations

from sagellm_benchmark.compatibility_matrix import (
    CompatibilityCase,
    CompatibilityMatrixConfig,
    evaluate_compatibility_matrix,
)


def test_compatibility_matrix_tracks_versions_errors_and_consistency() -> None:
    result = evaluate_compatibility_matrix(
        [
            CompatibilityCase(
                name="chat-nonstream-ok",
                protocol_version="v1",
                negotiated_version="v1",
                endpoint_type="chat",
                stream=False,
                expected_success=True,
                observed_success=True,
                response_text="hello world",
                consistency_group="hello",
            ),
            CompatibilityCase(
                name="chat-stream-ok",
                protocol_version="v1",
                negotiated_version="v1",
                endpoint_type="chat",
                stream=True,
                expected_success=True,
                observed_success=True,
                response_text="hello world",
                consistency_group="hello",
            ),
            CompatibilityCase(
                name="version-mismatch-fail",
                protocol_version="v2",
                negotiated_version="v1",
                endpoint_type="chat",
                stream=False,
                expected_success=False,
                observed_success=False,
                expected_error_code="invalid_argument",
                observed_error_code="invalid_argument",
            ),
        ]
    )

    assert result["overall_passed"] is True
    assert result["compatibility_pass_rate"] == 1.0
    assert result["version_mismatch_total"] == 1
    assert result["error_code_distribution"] == {"invalid_argument": 1}
    assert result["stream_nonstream_consistency"][0]["passed"] is True


def test_compatibility_matrix_detects_inconsistent_stream_nonstream_outputs() -> None:
    result = evaluate_compatibility_matrix(
        [
            CompatibilityCase(
                name="chat-nonstream-ok",
                protocol_version="v1",
                negotiated_version="v1",
                endpoint_type="chat",
                stream=False,
                expected_success=True,
                observed_success=True,
                response_text="hello world",
                consistency_group="hello",
            ),
            CompatibilityCase(
                name="chat-stream-drift",
                protocol_version="v1",
                negotiated_version="v1",
                endpoint_type="chat",
                stream=True,
                expected_success=True,
                observed_success=True,
                response_text="HELLO WORLD",
                consistency_group="hello",
            ),
        ]
    )

    assert result["overall_passed"] is False
    assert result["stream_nonstream_consistency"][0]["passed"] is False


def test_compatibility_matrix_config_reads_switches() -> None:
    config = CompatibilityMatrixConfig.from_env(
        {"SAGELLM_BENCH_COMPAT": "1", "SAGELLM_BENCH_COMPAT_KILL": "0"}
    )

    assert config.enabled is True
    assert config.kill_switch_active is False
