#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <endpoint_a> <endpoint_b> [model]"
  echo "Example: $0 http://127.0.0.1:8901/v1 http://127.0.0.1:8000/v1 Qwen/Qwen2.5-0.5B-Instruct"
  exit 1
fi

ENDPOINT_A="$1"
ENDPOINT_B="$2"
MODEL="${3:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_DIR="${OUT_DIR:-./benchmark_results/compare_$(date +%Y%m%d_%H%M%S)}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-64}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
BATCH_SIZES_CSV="${BATCH_SIZES:-1,2,4}"

mkdir -p "$OUT_DIR"

run_one() {
  local name="$1"
  local endpoint="$2"
  local json_file="$OUT_DIR/${name}.json"
  local md_file="$OUT_DIR/${name}.md"

    IFS=',' read -r -a batch_sizes <<< "$BATCH_SIZES_CSV"

    args=(
        perf
        --type e2e
        --live
        --backend-url "$endpoint"
        --model "$MODEL"
        --max-output-tokens "$MAX_OUTPUT_TOKENS"
        --request-timeout "$REQUEST_TIMEOUT"
        --output-json "$json_file"
        --output-markdown "$md_file"
    )

    for bs in "${batch_sizes[@]}"; do
        bs_trimmed="${bs//[[:space:]]/}"
        if [[ -n "$bs_trimmed" ]]; then
            args+=(--batch-size "$bs_trimmed")
        fi
    done

    echo "===== Running $name @ $endpoint (batch_sizes=$BATCH_SIZES_CSV) ====="
    sagellm-benchmark "${args[@]}"
}

run_one "endpoint_a" "$ENDPOINT_A"
run_one "endpoint_b" "$ENDPOINT_B"

python - "$OUT_DIR" "$ENDPOINT_A" "$ENDPOINT_B" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
endpoint_a = sys.argv[2]
endpoint_b = sys.argv[3]

def load_summary(path: pathlib.Path):
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    return {
        "rows": summary.get("total_rows", 0),
        "ttft_ms": float(summary.get("avg_ttft_ms", 0.0)),
        "tbt_ms": float(summary.get("avg_tbt_ms", 0.0)),
        "tps": float(summary.get("avg_throughput_tps", 0.0)),
    }

a = load_summary(out_dir / "endpoint_a.json")
b = load_summary(out_dir / "endpoint_b.json")

def ratio(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0

report = [
    "# Endpoint Comparison Summary",
    "",
    f"- endpoint_a: {endpoint_a}",
    f"- endpoint_b: {endpoint_b}",
    "",
    "| Metric | endpoint_a | endpoint_b | Delta(b-a) |",
    "|---|---:|---:|---:|",
    f"| Avg TTFT (ms) | {a['ttft_ms']:.2f} | {b['ttft_ms']:.2f} | {b['ttft_ms'] - a['ttft_ms']:+.2f} |",
    f"| Avg TBT (ms) | {a['tbt_ms']:.2f} | {b['tbt_ms']:.2f} | {b['tbt_ms'] - a['tbt_ms']:+.2f} |",
    f"| Avg TPS | {a['tps']:.2f} | {b['tps']:.2f} | {b['tps'] - a['tps']:+.2f} ({ratio(b['tps'], a['tps']):+.1f}%) |",
]

(out_dir / "comparison.md").write_text("\n".join(report) + "\n")
print("\n".join(report))
PY

echo ""
echo "Saved reports under: $OUT_DIR"
