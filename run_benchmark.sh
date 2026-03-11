#!/usr/bin/env bash
# Profile-aware benchmark runner for sageLLM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE="quick"
OUTPUT_DIR=""
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
REQUEST_TIMEOUT="180"
SERVER_WAIT_S="120"
MAX_OUTPUT_TOKENS="64"
WORKLOAD="all"
BACKEND="cpu"
UPLOAD_HF_MODE="auto"
PROMPT_CLEANUP=0
BATCH_SIZES_SET=0
declare -a BATCH_SIZES=("1" "2" "4")
declare -a TARGETS=()
declare -a TARGET_COMMANDS=()
declare -a LOG_FILES=()

SHARED_STREAM_MARKERS="${SHARED_STREAM_MARKERS:-shared_batched_stream_worker|cross_request_batching|SharedStreamBatchStart}"
PAGED_PATH_MARKERS="${PAGED_PATH_MARKERS:-block_tables|paged_attention|paged:implementation|native-ascend|native-kunlun|native-musa|torch-fallback}"
BLOCK_TABLE_MARKERS="${BLOCK_TABLE_MARKERS:-block_tables|has_block_tables|slot_mapping|context_lens}"

usage() {
    cat <<'EOF'
Usage:
  ./run_benchmark.sh [OUTPUT_DIR]
  ./run_benchmark.sh --profile quick [--output-dir DIR]
  ./run_benchmark.sh --profile convergence --target LABEL=URL --target LABEL=URL [options]

Profiles:
  quick        Run the default CPU Q1-Q8 suite and optional HF upload.
  convergence  Compare live OpenAI-compatible endpoints and capture validation artifacts.

Convergence options:
  --target LABEL=URL            Repeat at least twice.
  --target-command LABEL=CMD    Optional local start command for a target.
  --log-file LABEL=PATH         Optional service log for marker probes.
  --model MODEL                 Model name used by compare. Default: Qwen/Qwen2.5-0.5B-Instruct
  --batch-size N                Repeatable. Defaults to 1,2,4.
  --max-output-tokens N         Default: 64.
  --request-timeout SEC         Default: 180.
  --server-wait SEC             Default: 120.
  --prompt-cleanup              Forward cleanup prompt to compare.
  --no-prompt-cleanup           Disable cleanup prompt.

Quick options:
  --workload NAME               Default: all.
  --backend NAME                Default: cpu.
  --upload-hf auto|always|never Default: auto.

General options:
  --output-dir DIR
  -h, --help

Examples:
  ./run_benchmark.sh

  ./run_benchmark.sh --profile convergence \
    --target before=http://127.0.0.1:8901/v1 \
    --target after=http://127.0.0.1:8902/v1 \
    --log-file before=/tmp/sagellm-before.log \
    --log-file after=/tmp/sagellm-after.log \
    --model Qwen/Qwen2.5-0.5B-Instruct
EOF
}

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command not found: $cmd" >&2
        exit 1
    fi
}

slugify() {
    printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '-'
}

parse_label_mapping() {
    local spec="$1"
    if [[ "$spec" != *=* ]]; then
        echo "Error: expected LABEL=VALUE, got '$spec'" >&2
        exit 1
    fi
    printf '%s\n' "${spec%%=*}" "${spec#*=}"
}

root_url_from_v1() {
    local url="$1"
    url="${url%/}"
    if [[ "$url" == */v1 ]]; then
        printf '%s\n' "${url%/v1}"
    else
        printf '%s\n' "$url"
    fi
}

shell_join() {
    local out=""
    local part
    for part in "$@"; do
        printf -v part '%q' "$part"
        out+="${part} "
    done
    printf '%s\n' "${out% }"
}

fetch_endpoint_probe() {
    local label="$1"
    local url="$2"
    local output_dir="$3"
    local slug
    local root_url
    slug="$(slugify "$label")"
    root_url="$(root_url_from_v1 "$url")"

    if command -v curl >/dev/null 2>&1; then
        if curl -fsSL "${root_url}/info" -o "${output_dir}/${slug}_info.json"; then
            echo "Captured ${label} /info -> ${output_dir}/${slug}_info.json"
        else
            echo "Warning: failed to capture ${label} /info from ${root_url}/info" >&2
            if curl -fsSL "${url%/}/models" -o "${output_dir}/${slug}_models.json"; then
                echo "Captured ${label} /v1/models -> ${output_dir}/${slug}_models.json"
            else
                echo "Warning: failed to capture ${label} /v1/models from ${url%/}/models" >&2
            fi
        fi

        if curl -fsSL "${root_url}/metrics" -o "${output_dir}/${slug}_metrics.prom"; then
            echo "Captured ${label} /metrics -> ${output_dir}/${slug}_metrics.prom"
        else
            echo "Warning: failed to capture ${label} /metrics from ${root_url}/metrics" >&2
        fi
    else
        echo "Warning: curl not found; skipping /info and /metrics probes" >&2
    fi
}

collect_log_probe() {
    local label="$1"
    local path="$2"
    local output_dir="$3"
    local slug
    local shared_hits
    local paged_hits
    local block_hits
    slug="$(slugify "$label")"

    if [[ ! -f "$path" ]]; then
        echo "Warning: log file not found for ${label}: ${path}" >&2
        return 0
    fi

    shared_hits="$(grep -E -c "$SHARED_STREAM_MARKERS" "$path" || true)"
    paged_hits="$(grep -E -c "$PAGED_PATH_MARKERS" "$path" || true)"
    block_hits="$(grep -E -c "$BLOCK_TABLE_MARKERS" "$path" || true)"

    python3 - "$output_dir" "$slug" "$label" "$path" "$shared_hits" "$paged_hits" "$block_hits" \
        "$SHARED_STREAM_MARKERS" "$PAGED_PATH_MARKERS" "$BLOCK_TABLE_MARKERS" <<'PY'
import json
import sys
from pathlib import Path

output_dir, slug, label, path, shared_hits, paged_hits, block_hits, shared_pat, paged_pat, block_pat = sys.argv[1:]
payload = {
    "label": label,
    "log_file": path,
    "shared_stream_markers": {
        "pattern": shared_pat,
        "hits": int(shared_hits),
    },
    "paged_path_markers": {
        "pattern": paged_pat,
        "hits": int(paged_hits),
    },
    "block_table_markers": {
        "pattern": block_pat,
        "hits": int(block_hits),
    },
}
Path(output_dir, f"{slug}_log_probe.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

    echo "Captured ${label} log probe -> ${output_dir}/${slug}_log_probe.json"
}

write_reproduction_script() {
    local output_dir="$1"
    shift
    local compare_cmd=("$@")
    local file_path="${output_dir}/REPRODUCE.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        echo
        echo "# Reproduce the convergence comparison"
        shell_join "${compare_cmd[@]}"
        echo
        local spec label url root slug
        for spec in "${TARGETS[@]}"; do
            mapfile -t _parts < <(parse_label_mapping "$spec")
            label="${_parts[0]}"
            url="${_parts[1]}"
            root="$(root_url_from_v1 "$url")"
            slug="$(slugify "$label")"
            echo "curl -fsSL ${root}/info -o ${slug}_info.json"
            echo "curl -fsSL ${root}/metrics -o ${slug}_metrics.prom"
        done
    } >"$file_path"
    chmod +x "$file_path"
}

write_validation_summary() {
    local output_dir="$1"
    python3 - "$output_dir" "$SHARED_STREAM_MARKERS" "$PAGED_PATH_MARKERS" "$BLOCK_TABLE_MARKERS" <<'PY'
import json
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
shared_pattern = sys.argv[2]
paged_pattern = sys.argv[3]
block_pattern = sys.argv[4]
comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
targets = []


def _count_pattern(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    return len(re.findall(pattern, text, flags=re.IGNORECASE))

for row in comparison.get("targets", []):
    label = row["label"]
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in label).strip("-") or "target"
    target_path = output_dir / f"{slug}.json"
    target_payload = json.loads(target_path.read_text(encoding="utf-8")) if target_path.exists() else {}
    summary = target_payload.get("summary", {})
    log_probe_path = output_dir / f"{slug}_log_probe.json"
    log_probe = json.loads(log_probe_path.read_text(encoding="utf-8")) if log_probe_path.exists() else None
    info_path = output_dir / f"{slug}_info.json"
    metrics_path = output_dir / f"{slug}_metrics.prom"
    models_path = output_dir / f"{slug}_models.json"
    artifacts = {
        "target_json": str(target_path.name) if target_path.exists() else None,
        "target_markdown": f"{slug}.md" if (output_dir / f"{slug}.md").exists() else None,
        "info_json": str(info_path.name) if info_path.exists() else None,
        "models_json": str(models_path.name) if models_path.exists() else None,
        "metrics_prom": str(metrics_path.name) if metrics_path.exists() else None,
        "log_probe_json": str(log_probe_path.name) if log_probe_path.exists() else None,
    }
    probe_coverage = {
        "info": info_path.exists(),
        "models": models_path.exists(),
        "metrics": metrics_path.exists(),
        "log_probe": log_probe_path.exists(),
    }
    runtime_surface_markers = {
        "shared_stream_markers": {
            "pattern": shared_pattern,
            "hits": _count_pattern(info_path, shared_pattern)
            + _count_pattern(models_path, shared_pattern)
            + _count_pattern(metrics_path, shared_pattern),
        },
        "paged_path_markers": {
            "pattern": paged_pattern,
            "hits": _count_pattern(info_path, paged_pattern)
            + _count_pattern(models_path, paged_pattern)
            + _count_pattern(metrics_path, paged_pattern),
        },
        "block_table_markers": {
            "pattern": block_pattern,
            "hits": _count_pattern(info_path, block_pattern)
            + _count_pattern(models_path, block_pattern)
            + _count_pattern(metrics_path, block_pattern),
        },
    }
    evidence_gaps = []
    if not probe_coverage["info"] and not probe_coverage["models"]:
        evidence_gaps.append("missing_endpoint_identity_probe")
    if not probe_coverage["metrics"]:
        evidence_gaps.append("missing_metrics_probe")
    if not probe_coverage["log_probe"]:
        evidence_gaps.append("missing_log_probe")
    elif log_probe is not None:
        shared_hits = int(log_probe.get("shared_stream_markers", {}).get("hits", 0))
        paged_hits = int(log_probe.get("paged_path_markers", {}).get("hits", 0))
        block_hits = int(log_probe.get("block_table_markers", {}).get("hits", 0))
        if shared_hits == 0 and paged_hits == 0 and block_hits == 0:
            evidence_gaps.append("log_probe_marker_hits_zero")
    if (
        runtime_surface_markers["shared_stream_markers"]["hits"] == 0
        and runtime_surface_markers["paged_path_markers"]["hits"] == 0
        and runtime_surface_markers["block_table_markers"]["hits"] == 0
    ):
        evidence_gaps.append("runtime_surface_marker_hits_zero")
    targets.append(
        {
            "label": label,
            "url": row.get("url"),
            "summary": {
                "avg_ttft_ms": summary.get("avg_ttft_ms", row.get("avg_ttft_ms", 0.0)),
                "avg_tbt_ms": summary.get("avg_tbt_ms", row.get("avg_tbt_ms", 0.0)),
                "avg_throughput_tps": summary.get("avg_throughput_tps", row.get("avg_throughput_tps", 0.0)),
                "output_throughput_tps": summary.get("output_throughput_tps", 0.0),
                "request_throughput_rps": summary.get("request_throughput_rps", 0.0),
                "total_rows": summary.get("total_rows", row.get("rows", 0)),
            },
            "delta_vs_baseline": row.get("delta_vs_baseline", {}),
            "artifacts": artifacts,
            "probe_coverage": probe_coverage,
            "runtime_surface_markers": runtime_surface_markers,
            "evidence_gaps": evidence_gaps,
            "log_probe": log_probe,
        }
    )

payload = {
    "profile": "convergence",
    "baseline": comparison.get("baseline"),
    "model": comparison.get("model"),
    "batch_sizes": comparison.get("batch_sizes", []),
    "result_fields": [
        "avg_ttft_ms",
        "avg_tbt_ms",
        "avg_throughput_tps",
        "output_throughput_tps",
        "request_throughput_rps",
        "shared_stream_markers.hits",
        "paged_path_markers.hits",
        "block_table_markers.hits",
    ],
    "validation_checks": {
        "shared_stream_concurrent_batching": "Compare avg_ttft_ms, avg_tbt_ms, and output_throughput_tps at batch size >= 2, then inspect *_log_probe.json shared_stream_markers.hits.",
        "paged_or_native_path_usage": "Inspect *_metrics.prom, *_info.json, and *_log_probe.json paged_path_markers.hits for block_tables/native implementation evidence.",
        "formal_block_table_usage": "Inspect *_log_probe.json block_table_markers.hits and corresponding *_metrics.prom snapshots.",
    },
    "targets": targets,
}

(output_dir / "validation_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Convergence Validation Summary",
    "",
    f"- Baseline: {payload['baseline']}",
    f"- Model: {payload['model']}",
    f"- Batch sizes: {', '.join(str(item) for item in payload['batch_sizes'])}",
    "",
    "## Result Fields",
    "",
]
for field in payload["result_fields"]:
    lines.append(f"- {field}")

lines.extend(["", "## Targets", ""])
for target in targets:
    summary = target["summary"]
    probe_coverage = target.get("probe_coverage", {})
    runtime_surface_markers = target.get("runtime_surface_markers", {})
    evidence_gaps = target.get("evidence_gaps", [])
    lines.append(
        "- {label}: TTFT={ttft:.2f}ms, TBT={tbt:.2f}ms, Output TPS={otps:.2f}, "
        "SharedMarkers={shared}, PagedMarkers={paged}, BlockTableMarkers={block}".format(
            label=target["label"],
            ttft=float(summary.get("avg_ttft_ms", 0.0)),
            tbt=float(summary.get("avg_tbt_ms", 0.0)),
            otps=float(summary.get("output_throughput_tps", 0.0)),
            shared=(target.get("log_probe") or {}).get("shared_stream_markers", {}).get("hits", 0),
            paged=(target.get("log_probe") or {}).get("paged_path_markers", {}).get("hits", 0),
            block=(target.get("log_probe") or {}).get("block_table_markers", {}).get("hits", 0),
        )
    )
    lines.append(
        "  ProbeCoverage: info={info}, models={models}, metrics={metrics}, log_probe={log_probe}".format(
            info=probe_coverage.get("info", False),
            models=probe_coverage.get("models", False),
            metrics=probe_coverage.get("metrics", False),
            log_probe=probe_coverage.get("log_probe", False),
        )
    )
    lines.append(
        "  RuntimeSurfaceMarkers: shared={shared}, paged={paged}, block_tables={block}".format(
            shared=runtime_surface_markers.get("shared_stream_markers", {}).get("hits", 0),
            paged=runtime_surface_markers.get("paged_path_markers", {}).get("hits", 0),
            block=runtime_surface_markers.get("block_table_markers", {}).get("hits", 0),
        )
    )
    if evidence_gaps:
        lines.append("  EvidenceGaps: " + ", ".join(evidence_gaps))

(output_dir / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

run_quick_profile() {
    local output_dir="$1"

    echo "======================================"
    echo "  sageLLM Benchmark - Q1~Q8 Workloads"
    echo "======================================"
    echo
    echo "Output directory: $output_dir"
    echo

    require_command sagellm-benchmark

    echo "[1/5] Running ${WORKLOAD} workloads with ${BACKEND} backend..."
    sagellm-benchmark run --workload "$WORKLOAD" --backend "$BACKEND" --output "$output_dir" -v

    echo
    echo "[2/5] Generating summary report..."
    sagellm-benchmark report --input "$output_dir/benchmark_summary.json" --format table

    echo
    echo "[3/5] Generating markdown report..."
    sagellm-benchmark report --input "$output_dir/benchmark_summary.json" --format markdown >"$output_dir/REPORT.md"

    echo
    echo "[4/5] Results saved to:"
    echo "  - $output_dir/benchmark_summary.json"
    echo "  - $output_dir/Q1_metrics.json ... Q8_metrics.json"
    echo "  - $output_dir/Q1_leaderboard.json ... Q8_leaderboard.json"
    echo "  - $output_dir/REPORT.md"

    echo
    case "$UPLOAD_HF_MODE" in
        always)
            echo "[5/5] Uploading leaderboard data to Hugging Face..."
            sagellm-benchmark upload-hf \
                --input "$output_dir" \
                --dataset intellistream/sagellm-benchmark-results \
                --token "${HF_TOKEN:?HF_TOKEN must be set when --upload-hf always is used}"
            echo
            echo "✓ HF upload complete!"
            ;;
        auto)
            if [[ -n "${HF_TOKEN:-}" ]]; then
                echo "[5/5] Uploading leaderboard data to Hugging Face..."
                sagellm-benchmark upload-hf \
                    --input "$output_dir" \
                    --dataset intellistream/sagellm-benchmark-results \
                    --token "$HF_TOKEN"
                echo
                echo "✓ HF upload complete!"
            else
                echo "[5/5] Skipping HF upload (HF_TOKEN not set)"
                echo "      To enable: export HF_TOKEN=hf_xxx && ./run_benchmark.sh"
            fi
            ;;
        never)
            echo "[5/5] Skipping HF upload (--upload-hf never)"
            ;;
        *)
            echo "Error: unsupported --upload-hf mode: $UPLOAD_HF_MODE" >&2
            exit 1
            ;;
    esac

    echo
    echo "✓ Benchmark completed successfully!"
}

run_convergence_profile() {
    local output_dir="$1"
    local compare_cmd
    local prompt_flag="--no-prompt-cleanup"
    local spec label url log_spec log_label log_path

    require_command sagellm-benchmark
    require_command python3

    if [[ "${#TARGETS[@]}" -lt 2 ]]; then
        echo "Error: --profile convergence requires at least two --target LABEL=URL entries" >&2
        exit 1
    fi

    mkdir -p "$output_dir"

    compare_cmd=(sagellm-benchmark compare)
    for spec in "${TARGETS[@]}"; do
        compare_cmd+=(--target "$spec")
    done
    for spec in "${TARGET_COMMANDS[@]}"; do
        compare_cmd+=(--target-command "$spec")
    done
    for spec in "${BATCH_SIZES[@]}"; do
        compare_cmd+=(--batch-size "$spec")
    done
    if [[ "$PROMPT_CLEANUP" -eq 1 ]]; then
        prompt_flag="--prompt-cleanup"
    fi
    compare_cmd+=(
        --model "$MODEL"
        --request-timeout "$REQUEST_TIMEOUT"
        --server-wait "$SERVER_WAIT_S"
        --max-output-tokens "$MAX_OUTPUT_TOKENS"
        --output-dir "$output_dir"
        "$prompt_flag"
    )

    echo "==============================================="
    echo "  sageLLM Benchmark - Convergence Validation"
    echo "==============================================="
    echo
    echo "Output directory: $output_dir"
    echo "Model: $MODEL"
    echo "Batch sizes: ${BATCH_SIZES[*]}"
    echo

    write_reproduction_script "$output_dir" "${compare_cmd[@]}"

    echo "[1/4] Running live compare profile..."
    "${compare_cmd[@]}"

    echo
    echo "[2/4] Capturing /info and /metrics probes..."
    for spec in "${TARGETS[@]}"; do
        mapfile -t _parts < <(parse_label_mapping "$spec")
        label="${_parts[0]}"
        url="${_parts[1]}"
        fetch_endpoint_probe "$label" "$url" "$output_dir"
    done

    echo
    echo "[3/4] Capturing optional log probes..."
    for log_spec in "${LOG_FILES[@]}"; do
        mapfile -t _log_parts < <(parse_label_mapping "$log_spec")
        log_label="${_log_parts[0]}"
        log_path="${_log_parts[1]}"
        collect_log_probe "$log_label" "$log_path" "$output_dir"
    done

    echo
    echo "[4/4] Building validation summary..."
    write_validation_summary "$output_dir"

    echo
    echo "Artifacts saved to:"
    echo "  - $output_dir/comparison.json"
    echo "  - $output_dir/comparison.md"
    echo "  - $output_dir/validation_summary.json"
    echo "  - $output_dir/VALIDATION.md"
    echo "  - $output_dir/REPRODUCE.sh"
    echo "  - $output_dir/*_info.json"
    echo "  - $output_dir/*_metrics.prom"
    echo "  - $output_dir/*_log_probe.json"
    echo
    echo "✓ Convergence validation completed successfully!"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --target)
            TARGETS+=("$2")
            shift 2
            ;;
        --target-command)
            TARGET_COMMANDS+=("$2")
            shift 2
            ;;
        --log-file)
            LOG_FILES+=("$2")
            shift 2
            ;;
        --batch-size)
            if [[ "$BATCH_SIZES_SET" -eq 0 ]]; then
                BATCH_SIZES=()
                BATCH_SIZES_SET=1
            fi
            BATCH_SIZES+=("$2")
            shift 2
            ;;
        --max-output-tokens)
            MAX_OUTPUT_TOKENS="$2"
            shift 2
            ;;
        --request-timeout)
            REQUEST_TIMEOUT="$2"
            shift 2
            ;;
        --server-wait)
            SERVER_WAIT_S="$2"
            shift 2
            ;;
        --workload)
            WORKLOAD="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --upload-hf)
            UPLOAD_HF_MODE="$2"
            shift 2
            ;;
        --prompt-cleanup)
            PROMPT_CLEANUP=1
            shift
            ;;
        --no-prompt-cleanup)
            PROMPT_CLEANUP=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            if [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
                shift
            else
                echo "Error: unexpected argument: $1" >&2
                usage >&2
                exit 1
            fi
            ;;
    esac
done

case "$PROFILE" in
    quick)
        OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"
        run_quick_profile "$OUTPUT_DIR"
        ;;
    convergence)
        OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results/convergence_$(date +%Y%m%d_%H%M%S)}"
        run_convergence_profile "$OUTPUT_DIR"
        ;;
    *)
        echo "Error: unsupported profile '$PROFILE'" >&2
        usage >&2
        exit 1
        ;;
esac
