# Canonical Benchmark Result Schema

> Status: active direction, implemented incrementally in current CLI
> Scope: benchmark artifact unification across `run`, `compare`, shell compatibility wrappers, parity, runtime validation, telemetry, and compatibility leaderboard export.

## 1. Purpose

`sagellm-benchmark` now treats canonical artifacts as the only internal benchmark source of truth. Execution entrypoints write canonical artifacts first, and leaderboard-facing JSON is generated only afterward as a compatibility export boundary.

Current responsibilities:

- `run`: canonical local workload benchmark pipeline
- `compare`: canonical live multi-endpoint benchmark pipeline
- `run_benchmark.sh --profile quick`: compatibility wrapper over `run`
- `run_benchmark.sh --profile convergence`: compatibility wrapper over `compare` plus extra probe packaging

This document defines why all of those paths converge on one canonical result model instead of keeping a separate leaderboard-first benchmark branch.

`sagellm-benchmark` still emits multiple user-visible files:

- local run outputs (`config.json`, `benchmark_summary.json`, `*_metrics.json`, `*_leaderboard.json`)
- live endpoint compare outputs (`<label>.json`, `comparison.json`, `comparison.md`)
- convergence evidence (`validation_summary.json`, `VALIDATION.md`, `*_info.json`, `*_metrics.prom`, `*_log_probe.json`)
- parity artifacts (`<label>.parity.json`)
- live runtime consistency reports (`*_runtime_consistency.json`)
- core decode telemetry artifacts (`*_core_telemetry.json`)

These artifacts overlap in semantics but not in structure. The result used to be duplicated merge logic, duplicated exporter logic, and frontend-side field guessing.

The design rule is now explicit: all benchmark entrypoints emit canonical artifacts first. Leaderboard JSON is a pure compatibility export derived from canonical artifacts rather than a parallel source of truth.

## 2. Design Goals

The canonical schema must:

- cover local workload runs and live endpoint compares with one envelope
- support CPU, CUDA/GPU, Ascend/NPU, and future hardware without schema forks
- keep compare evidence, validation evidence, telemetry, and leaderboard export fields in one model
- fail fast when required data for an artifact kind is missing
- avoid silent default filling for required semantics
- allow a single execution result to be promoted into leaderboard export when policy allows it
- keep `compare` as the only live benchmark mainline instead of reintroducing a leaderboard-first side path
- keep benchmark behavior chat-first and avoid a completions-only benchmark fork

## 3. Entrypoint and Compatibility Rules

### 3.1 Mainline entrypoints

- `run` is the mainline local workload runner.
- `compare` is the mainline live endpoint compare runner.

### 3.2 Setup helpers

- `vllm-compare install-ascend` remains an environment/bootstrap helper for validated Ascend compare setup.

### 3.3 Compatibility wrappers

- `quick` and `convergence` from `run_benchmark.sh` are compatibility shell wrappers.
- `compare-record` and `compare-offline` are compatibility capture/report helpers for constrained workflows.
- `upload-hf` is a compatibility publish helper over the derived leaderboard export boundary. Prefer `--publish` on mainline benchmark commands.

### 3.4 Compatibility export boundary

The following files are compatibility outputs only:

- `*_leaderboard.json`
- `leaderboard_manifest.json`
- `leaderboard_single.json`
- `leaderboard_multi.json`
- `last_updated.json`

No benchmark execution path should depend on those files to complete measurement or comparison logic.

## 4. Current Artifact Survey

### 4.1 Existing local run outputs

Current local run flow spreads semantics across:

- `config.json`: invocation, model, backend, versions
- `benchmark_summary.json`: run-level workload summary
- per-workload metrics JSON files: workload measurements
- `*_leaderboard.json`: website-oriented export rows

Problems:

- no single canonical row for one workload execution
- leaderboard export duplicates execution metadata instead of referencing canonical execution data
- workload suite context and per-workload result context are split across files

### 4.2 Existing compare outputs

Current `compare` / `compare-record` emits per target:

- `kind = e2e`
- `label`, `url`, `hardware_family`, `models`, `batch_sizes`, `precisions`
- `summary`, `rows`, `runtime_artifacts`

And emits one compare summary:

- `kind = compare`
- `model`, `batch_sizes`, `baseline`, `targets[*].delta_vs_baseline`

Problems:

- compare target artifacts and local run artifacts do not share one schema
- compare summary has no typed reference to validation, parity, or telemetry findings

### 4.3 Existing validation and telemetry outputs

Current specialized outputs include:

- `parity-run/v1`
- `live-runtime-consistency/v1`
- `core-decode-step-telemetry/v1`
- convergence `validation_summary.json`

Problems:

- each artifact has its own envelope and lifecycle
- the same execution context is repeated in multiple files
- there is no stable way to say "these validation and telemetry facts belong to this execution result"

### 4.4 Existing leaderboard outputs

Current website schema expects one flattened entry with:

- engine and version
- hardware and environment
- model and workload
- summary metrics
- metadata and reproducibility fields

Problems:

- exporter must infer fields from multiple upstream files
- some fields are reverse-engineered by frontend logic, such as workload ID from `metadata.notes`

## 5. Canonical Schema Overview

The canonical format is a single JSON envelope with a required `artifact_kind` discriminator.

All artifacts share one base structure:

```json
{
  "schema_version": "canonical-benchmark-result/v2",
  "artifact_kind": "execution_result",
  "artifact_id": "uuid",
  "produced_at": "2026-03-14T12:00:00Z",
  "producer": { ... },
  "provenance": { ... },
  "hardware": { ... },
  "engine": { ... },
  "versions": { ... },
  "workload": { ... },
  "metrics": { ... },
  "validation": { ... },
  "telemetry": { ... },
  "artifacts": { ... },
  "relations": { ... },
  "leaderboard": { ... }
}
```

The artifact kinds are:

- `execution_result`: one concrete workload execution on one engine and one hardware target
- `comparison_result`: one comparison over multiple `execution_result` artifacts
- `run_manifest`: one local benchmark run that groups multiple `execution_result` artifacts
- `validation_report`: one validation-only artifact attached to one `execution_result`
- `telemetry_capture`: one telemetry-only artifact attached to one `execution_result`

The schema remains singular because each artifact kind uses the same envelope and relation model.

## 6. Base Envelope

### 5.1 Required top-level fields for every artifact

```json
{
  "schema_version": "canonical-benchmark-result/v2",
  "artifact_kind": "execution_result|comparison_result|run_manifest|validation_report|telemetry_capture",
  "artifact_id": "uuid",
  "produced_at": "date-time",
  "producer": {
    "tool": "sagellm-benchmark",
    "tool_version": "string",
    "command": "string",
    "subcommand": "run|compare|compare-record|compare-offline|validate-serving-consistency|perf",
    "host": "string"
  },
  "provenance": {
    "run_id": "string",
    "source_mode": "local-run|live-compare|offline-compare|convergence|validation|telemetry",
    "simulate": false,
    "git": {
      "repo": "string",
      "commit": "string|null",
      "dirty": false
    },
    "parents": ["artifact_id"],
    "tags": ["string"]
  }
}
```

Fail-fast rules:

- all base envelope fields above are required for all artifact kinds
- `artifact_id` must be stable UUID text, not a filename surrogate
- `producer.subcommand` must reflect the actual producing entrypoint
- `provenance.source_mode` is required and must not be inferred later by exporters

## 7. Required Sections by Artifact Kind

| Section | execution_result | compare_summary | run_manifest | validation_report | telemetry_capture |
|---|---|---|---|---|---|
| `hardware` | required | optional | optional | optional | optional |
| `engine` | required | optional | optional | optional | optional |
| `versions` | required | optional | optional | optional | optional |
| `workload` | required | optional | optional | optional | optional |
| `metrics` | required | optional | optional | optional | optional |
| `validation` | required | required | optional | required | optional |
| `telemetry` | required | optional | optional | optional | required |
| `artifacts` | required | required | required | required | required |
| `relations` | required | required | required | required | required |
| `leaderboard` | compatibility export only | forbidden | forbidden | forbidden | forbidden |

Fail-fast rules:

- `execution_result` without `hardware`, `engine`, `versions`, `workload`, `metrics`, `relations`, and `leaderboard` is invalid
- `comparison_result` must not pretend to be leaderboard-exportable
- `validation_report` and `telemetry_capture` must reference exactly one parent `execution_result`

## 8. Section Definitions

### 7.1 `hardware`

```json
{
  "hardware": {
    "family": "cpu|cuda|ascend|rocm|musa|xpu|other",
    "vendor": "string",
    "chip_model": "string",
    "chip_count": 1,
    "chips_per_node": 1,
    "node_count": 1,
    "interconnect": "None|NVLink|HCCS|PCIe|Ethernet|InfiniBand|custom",
    "memory_per_chip_gb": 80.0,
    "total_memory_gb": 80.0,
    "device_api": "cpu|cuda|npu|hip|xpu|other",
    "driver": {
      "name": "nvidia|cann|rocm|none|other",
      "version": "string|null"
    }
  }
}
```

Fail-fast rules:

- `family`, `vendor`, `chip_model`, `chip_count`, `node_count`, and `device_api` are required on `execution_result`
- `chip_count >= 1`, `node_count >= 1`
- when `node_count > 1`, `interconnect` must not be `None`

### 7.2 `engine`

```json
{
  "engine": {
    "name": "sagellm|vllm|vllm-ascend|lmdeploy|sglang|other",
    "version": "string",
    "engine_family": "native|third-party",
    "backend": "cpu|cuda|ascend|rocm|other",
    "serving_mode": "embedded|local-endpoint|remote-endpoint",
    "endpoint": {
      "base_url": "string|null",
      "health_checked": true
    }
  }
}
```

Fail-fast rules:

- `name`, `version`, `backend`, and `serving_mode` are required on `execution_result`
- `endpoint.base_url` is required when `serving_mode != embedded`

### 7.3 `versions`

```json
{
  "versions": {
    "sagellm": "string|null",
    "benchmark": "string",
    "protocol": "string|null",
    "backend": "string|null",
    "core": "string|null",
    "control_plane": "string|null",
    "gateway": "string|null",
    "kv_cache": "string|null",
    "comm": "string|null",
    "compression": "string|null",
    "torch": "string|null",
    "transformers": "string|null",
    "torch_npu": "string|null",
    "vllm": "string|null",
    "lmdeploy": "string|null",
    "python": "string",
    "os": "string"
  }
}
```

Fail-fast rules:

- `benchmark`, `python`, and `os` are required on `execution_result`
- component version may be `null` only if the component is truly not in the runtime stack
- exporters must not rewrite `null` to `N/A`; that transformation belongs to export-only logic

### 7.4 `workload`

```json
{
  "workload": {
    "suite_id": "q1-q8|perf-e2e|perf-operator|nonstream-compare|custom",
    "workload_id": "Q1|Q2|...|custom",
    "scenario_id": "string",
    "prompt_tokens": 128,
    "output_tokens": 128,
    "batch_size": 4,
    "concurrency": 4,
    "request_count": 32,
    "precision": "FP32|FP16|BF16|INT8|INT4|FP8|live",
    "dataset": {
      "name": "string|null",
      "split": "string|null",
      "source": "synthetic|sharegpt|scbench|custom|null"
    },
    "streaming": false,
    "simulate": false
  }
}
```

Fail-fast rules:

- `suite_id`, `workload_id`, `scenario_id`, `batch_size`, `concurrency`, and `precision` are required on `execution_result`
- Q1-Q8 workloads must carry explicit `workload_id`; frontend must not parse this from notes

### 7.5 `metrics`

```json
{
  "metrics": {
    "summary": {
      "total_requests": 32,
      "successful_requests": 32,
      "failed_requests": 0,
      "error_rate": 0.0,
      "avg_ttft_ms": 12.3,
      "p50_ttft_ms": 12.0,
      "p95_ttft_ms": 15.2,
      "p99_ttft_ms": 18.9,
      "avg_tbt_ms": 2.5,
      "avg_tpot_ms": 3.1,
      "avg_itl_ms": 2.4,
      "avg_e2el_ms": 210.0,
      "avg_throughput_tps": 88.0,
      "total_throughput_tps": 320.0,
      "request_throughput_rps": 12.0,
      "input_throughput_tps": 1536.0,
      "output_throughput_tps": 1056.0,
      "peak_mem_mb": 12345,
      "total_input_tokens": 4096,
      "total_output_tokens": 2816,
      "total_kv_used_tokens": 8192,
      "total_kv_used_bytes": 1048576,
      "avg_prefix_hit_rate": 0.0,
      "total_evict_count": 0,
      "total_evict_ms": 0.0,
      "avg_spec_accept_rate": 0.0,
      "total_time_s": 2.67
    },
    "rows": [
      {
        "scenario_id": "Q1_b4",
        "batch_size": 4,
        "successful_requests": 4,
        "failed_requests": 0,
        "avg_ttft_ms": 12.3,
        "avg_tbt_ms": 2.5,
        "output_throughput_tps": 1056.0
      }
    ]
  }
}
```

Fail-fast rules:

- `metrics.summary` is required on `execution_result`
- `avg_ttft_ms`, `avg_throughput_tps`, `peak_mem_mb`, `error_rate`, `successful_requests`, `failed_requests`, `total_requests` are required in `summary`
- compare and perf flows that have per-row data must preserve that data in `metrics.rows`; they must not only keep aggregated summary

### 7.6 `validation`

```json
{
  "validation": {
    "publishable_to_leaderboard": true,
    "parity": {
      "present": true,
      "schema_version": "parity-run/v1",
      "gate_id": "cuda_decode_parity_v1|null",
      "passed": true,
      "findings": []
    },
    "runtime_consistency": {
      "present": true,
      "schema_version": "live-runtime-consistency/v1",
      "passed": true,
      "findings": []
    },
    "evidence": {
      "endpoint_identity_probe": true,
      "metrics_probe": true,
      "log_probe": false,
      "runtime_surface_markers": {
        "shared_stream_hits": 0,
        "paged_path_hits": 1,
        "block_table_hits": 1
      },
      "evidence_gaps": []
    }
  }
}
```

Fail-fast rules:

- `validation.publishable_to_leaderboard` is required on `execution_result`
- leaderboard exporter must only accept artifacts with `publishable_to_leaderboard = true`
- if a command does not run parity or runtime consistency, it must explicitly record `present = false`; exporter must not invent pass status

### 7.7 `telemetry`

```json
{
  "telemetry": {
    "runtime_info": {
      "present": true,
      "path": "label_info.json|null",
      "summary": {
        "attention_selected_implementation": "native-cuda",
        "attention_selected_operator_pack": "flash-attn",
        "adjacent_selected_implementation": "native-cuda",
        "adjacent_selected_operator_pack": "decode-pack"
      }
    },
    "core_decode": {
      "present": true,
      "schema_version": "core-decode-step-telemetry/v1",
      "step_records": 128,
      "batch_sizes": [1, 2, 4],
      "selected_implementations": ["native-cuda"],
      "selected_operator_packs": ["decode-pack"]
    },
    "log_probe": {
      "present": false,
      "shared_stream_hits": 0,
      "paged_path_hits": 0,
      "block_table_hits": 0
    }
  }
}
```

Fail-fast rules:

- `telemetry` is required on `execution_result`, but inner captures may explicitly be absent
- absent telemetry must be represented as explicit `present = false`, not omitted

### 7.8 `artifacts`

```json
{
  "artifacts": {
    "primary_json": "string",
    "markdown": "string|null",
    "supporting": [
      {"kind": "parity", "path": "label.parity.json"},
      {"kind": "runtime_consistency", "path": "label_runtime_consistency.json"},
      {"kind": "core_telemetry", "path": "label_core_telemetry.json"}
    ]
  }
}
```

Fail-fast rules:

- `primary_json` is required for every canonical artifact
- each supporting artifact path must correspond to a section that says `present = true`

### 7.9 `relations`

```json
{
  "relations": {
    "parent_artifact_ids": ["uuid"],
    "child_artifact_ids": ["uuid"],
    "compare_group_id": "string|null",
    "baseline_artifact_id": "uuid|null",
    "target_artifact_ids": ["uuid"],
    "manifest_artifact_id": "uuid|null"
  }
}
```

Fail-fast rules:

- `compare_summary` requires `target_artifact_ids` with length `>= 2`
- `validation_report` and `telemetry_capture` require exactly one parent execution result in `parent_artifact_ids`

### 7.10 `leaderboard`

```json
{
  "leaderboard": {
    "entry_id": "uuid",
    "exportable": true,
    "config_type": "single_gpu|multi_gpu|multi_node",
    "reproducible_cmd": "string",
    "submitter": "string",
    "release_date": "YYYY-MM-DD|null",
    "changelog_url": "string|null",
    "notes": "string|null"
  }
}
```

Fail-fast rules:

- `leaderboard.exportable` and `leaderboard.entry_id` are required on `execution_result`
- compare summary and validation-only artifacts must not carry a `leaderboard` section

## 8. Canonical Emission Rules by Command

### 8.1 `run`

Emit:

- one `run_manifest`
- one `execution_result` per workload result

Do not emit leaderboard JSON directly from runner internals. Export leaderboard rows only from canonical `execution_result` artifacts after validation.

### 8.2 `compare` and `compare-record`

Emit:

- one `execution_result` per target
- one `compare_summary`

If parity or runtime evidence exists, either embed the summary into the `execution_result` or emit attached `validation_report` / `telemetry_capture` artifacts with typed relations.

### 8.3 `convergence`

Emit:

- one `execution_result` per target
- one `compare_summary`
- optional `validation_report` for gate or runtime consistency failures
- optional `telemetry_capture` for log probes and runtime probes

The convergence profile must not invent a second data model.

## 9. Leaderboard Export Contract

Leaderboard export should accept only canonical `execution_result` artifacts and perform a deterministic mapping.

Export eligibility requires:

- `artifact_kind = execution_result`
- `leaderboard.exportable = true`
- `validation.publishable_to_leaderboard = true`
- `hardware`, `engine`, `versions`, `workload`, and `metrics.summary` all present

Exporters must fail fast if any required leaderboard field cannot be derived without guessing.

## 10. Field Mapping From Current Outputs

### 10.1 Current `run` outputs to canonical `execution_result`

| Current source | Canonical field |
|---|---|
| `config.json.run_id` | `provenance.run_id` |
| `config.json.timestamp` | `produced_at` |
| `config.json.backend` | `engine.backend`, `hardware.family` |
| `config.json.model_path` | `workload.scenario_id` input context and `leaderboard.reproducible_cmd` source |
| `config.json.versions.*` | `versions.*` |
| per-workload `AggregatedMetrics` | `metrics.summary.*` |
| workload spec mapping Q1-Q8 | `workload.suite_id`, `workload.workload_id`, `workload.prompt_tokens`, `workload.output_tokens` |
| hardware/environment detector | `hardware.*`, `versions.python`, `versions.os`, runtime stack versions |

### 10.2 Current compare `e2e` payload to canonical `execution_result`

| Current source | Canonical field |
|---|---|
| `kind = e2e` | `artifact_kind = execution_result` |
| `label` | `engine.name` or `provenance.tags[]` depending on semantic meaning |
| `url` | `engine.endpoint.base_url` |
| `hardware_family` | `hardware.family` |
| `models[0]` | workload/model context used by exporter |
| `batch_sizes` | `workload.batch_size` plus `metrics.rows[*].batch_size` |
| `summary.*` | `metrics.summary.*` |
| `rows[*]` | `metrics.rows[*]` |
| `runtime_artifacts` | `artifacts.supporting[]`, `telemetry.*` |

### 10.3 Current `comparison.json` to canonical `compare_summary`

| Current source | Canonical field |
|---|---|
| `kind = compare` | `artifact_kind = compare_summary` |
| `baseline` | `relations.baseline_artifact_id` after artifact resolution |
| `targets[*].label` | `relations.target_artifact_ids[]` via artifact lookup |
| `targets[*].delta_vs_baseline.*` | `metrics.rows[*].comparison.delta_vs_baseline.*` or `validation.evidence` summary for compare artifact |

### 10.4 Current parity artifact to canonical `validation`

| Current source | Canonical field |
|---|---|
| `schema_version = parity-run/v1` | `validation.parity.schema_version` |
| `scenarios[*]` | `validation.parity.scenario_results[*]` |
| gate evaluation results | `validation.parity.passed`, `validation.parity.findings[]` |

### 10.5 Current runtime consistency report to canonical `validation`

| Current source | Canonical field |
|---|---|
| `schema_version = live-runtime-consistency/v1` | `validation.runtime_consistency.schema_version` |
| `passed` | `validation.runtime_consistency.passed` |
| `findings[]` | `validation.runtime_consistency.findings[]` |
| `observed` / `reference` | `validation.runtime_consistency.observed`, `validation.runtime_consistency.reference` |

### 10.6 Current core telemetry artifact to canonical `telemetry`

| Current source | Canonical field |
|---|---|
| `schema_version = core-decode-step-telemetry/v1` | `telemetry.core_decode.schema_version` |
| `summary.step_records` | `telemetry.core_decode.step_records` |
| `summary.batch_sizes` | `telemetry.core_decode.batch_sizes` |
| `summary.selected_implementations` | `telemetry.core_decode.selected_implementations` |
| `summary.selected_operator_packs` | `telemetry.core_decode.selected_operator_packs` |

### 10.7 Canonical `execution_result` to current leaderboard schema

| Canonical field | Leaderboard field |
|---|---|
| `leaderboard.entry_id` | `entry_id` |
| `engine.name` | `engine` |
| `engine.version` | `engine_version` |
| `versions.sagellm` | `sagellm_version` |
| `leaderboard.config_type` | `config_type` |
| `hardware.*` | `hardware.*` |
| `workload` + model context | `model.*`, `workload.*` |
| `metrics.summary.avg_ttft_ms` | `metrics.ttft_ms` |
| `metrics.summary.avg_tbt_ms` | `metrics.tbt_ms` |
| `metrics.summary.avg_tpot_ms` | `metrics.tpot_ms` |
| `metrics.summary.avg_throughput_tps` | `metrics.throughput_tps` |
| `metrics.summary.peak_mem_mb` | `metrics.peak_mem_mb` |
| `metrics.summary.error_rate` | `metrics.error_rate` |
| `metrics.summary.avg_prefix_hit_rate` | `metrics.prefix_hit_rate` |
| `metrics.summary.total_kv_used_tokens` | `metrics.kv_used_tokens` |
| `metrics.summary.total_kv_used_bytes` | `metrics.kv_used_bytes` |
| `metrics.summary.total_evict_count` | `metrics.evict_count` |
| `metrics.summary.total_evict_ms` | `metrics.evict_ms` |
| `metrics.summary.avg_spec_accept_rate` | `metrics.spec_accept_rate` |
| `versions.*` | `versions.*` |
| `leaderboard.reproducible_cmd` | `metadata.reproducible_cmd` |
| `leaderboard.submitter` | `metadata.submitter` |
| `leaderboard.release_date` | `metadata.release_date` |
| `leaderboard.changelog_url` | `metadata.changelog_url` |
| `leaderboard.notes` | `metadata.notes` |

## 11. Migration Plan

1. Introduce canonical artifact builders without changing website export yet.
2. Make `run` emit `run_manifest` plus per-workload `execution_result`.
3. Make `compare` emit per-target `execution_result` plus one `compare_summary`.
4. Attach parity, runtime consistency, and core telemetry through canonical `validation` and `telemetry` sections.
5. Rewrite leaderboard exporter to consume only canonical `execution_result` artifacts.
6. Deprecate direct leaderboard-first and ad hoc compare-first JSON shapes after one compatibility window.

## 12. Non-Goals

- No frontend schema is defined here.
- No Hugging Face dataset layout is defined here.
- No silent compatibility shim is required forever. Compatibility readers may exist temporarily, but canonical emission becomes mandatory.

## 13. Implementation Guardrails

- No fallback spaghetti. If an `execution_result` is missing required `engine`, `hardware`, `workload`, or `metrics.summary` semantics, artifact creation must fail.
- Use explicit `present = false` markers for optional evidence capture, not missing keys.
- Do not encode workload identity in free-form notes.
- Do not let website or upload logic infer benchmark semantics that benchmark can emit directly.
- Keep third-party compare ownership in `sagellm-benchmark`; do not move compare-specific schema concerns into `sagellm-core`.
