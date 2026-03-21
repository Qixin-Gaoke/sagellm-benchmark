"""sagellm-benchmark: Benchmark Suite & E2E Testing for sageLLM."""

from __future__ import annotations

from sagellm_benchmark._version import __version__

# Clients - Task B 客户端
from sagellm_benchmark.clients import (
    BenchmarkClient,
)
from sagellm_benchmark.compatibility_matrix import (
    CompatibilityCase,
    CompatibilityMatrixConfig,
    evaluate_compatibility_matrix,
)
from sagellm_benchmark.core_telemetry import (
    CoreDecodeStepTelemetryRecord,
    CoreDecodeTelemetryArtifact,
    CoreDecodeTelemetrySummary,
    build_core_decode_telemetry_artifact,
    extract_explicit_decode_snapshot,
    load_core_decode_telemetry_input,
)
from sagellm_benchmark.parity_gate import (
    DecodeParityGate,
    DecodeParityScenario,
    DecodeParityThresholds,
    GateFailureCategory,
    ParityGateEvaluation,
    ParityRunArtifact,
    ParityScenarioMetrics,
    build_default_cuda_decode_gate,
    evaluate_parity_gate,
)

# Traffic - 流量控制
from sagellm_benchmark.traffic import (
    ArrivalPattern,
    RampUpStrategy,
    RequestGenerator,
    TrafficController,
    TrafficProfile,
)

# Types - 公共数据类型（契约定义）
from sagellm_benchmark.types import (
    AggregatedMetrics,
    BenchmarkRequest,
    BenchmarkResult,
    ContractResult,
    ContractVersion,
    WorkloadSpec,
    WorkloadType,
)

__all__ = [
    "__version__",
    "CompatibilityCase",
    "CompatibilityMatrixConfig",
    "evaluate_compatibility_matrix",
    # Types (契约定义)
    "BenchmarkRequest",
    "BenchmarkResult",
    "WorkloadSpec",
    "WorkloadType",
    "AggregatedMetrics",
    "ContractResult",
    "ContractVersion",
    # Clients
    "BenchmarkClient",
    "CoreDecodeStepTelemetryRecord",
    "CoreDecodeTelemetryArtifact",
    "CoreDecodeTelemetrySummary",
    "build_core_decode_telemetry_artifact",
    "extract_explicit_decode_snapshot",
    "load_core_decode_telemetry_input",
    "DecodeParityGate",
    "DecodeParityScenario",
    "DecodeParityThresholds",
    "GateFailureCategory",
    "ParityRunArtifact",
    "ParityScenarioMetrics",
    "ParityGateEvaluation",
    "build_default_cuda_decode_gate",
    "evaluate_parity_gate",
    # Traffic
    "ArrivalPattern",
    "RampUpStrategy",
    "TrafficProfile",
    "RequestGenerator",
    "TrafficController",
]
