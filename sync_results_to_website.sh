#!/bin/bash
# Offline compatibility sync: standard benchmark leaderboard exports -> website snapshot files.
# Usage: ./sync_results_to_website.sh [website_repo_path] [benchmark_output_dir]

set -euo pipefail

WEBSITE_REPO=${1:-"../sagellm-website"}
BENCHMARK_OUTPUT_DIR=${2:-"outputs"}
BENCHMARK_REPO=$(pwd)

echo "==================================="
echo "Benchmark Results -> Website Sync"
echo "==================================="
echo "Benchmark repo:        $BENCHMARK_REPO"
echo "Benchmark output dir:  $BENCHMARK_OUTPUT_DIR"
echo "Website repo:          $WEBSITE_REPO"
echo ""

if [ ! -d "$WEBSITE_REPO" ]; then
    echo "❌ Website repository not found at: $WEBSITE_REPO"
    exit 1
fi

if [ ! -d "$BENCHMARK_OUTPUT_DIR" ]; then
    echo "❌ Benchmark output directory not found at: $BENCHMARK_OUTPUT_DIR"
    exit 1
fi

AGGREGATE_SCRIPT="$WEBSITE_REPO/scripts/aggregate_results.py"
if [ ! -f "$AGGREGATE_SCRIPT" ]; then
    echo "❌ Website aggregate script not found at: $AGGREGATE_SCRIPT"
    exit 1
fi

echo "🔍 Aggregating standard leaderboard exports..."
python3 "$AGGREGATE_SCRIPT" \
    --source-dir "$BENCHMARK_OUTPUT_DIR" \
    --output-dir "$WEBSITE_REPO/data"

echo ""
echo "==================================="
echo "✅ Sync completed"
echo "==================================="
echo "Files updated:"
echo "  - $WEBSITE_REPO/data/leaderboard_single.json"
echo "  - $WEBSITE_REPO/data/leaderboard_multi.json"
echo "  - $WEBSITE_REPO/data/leaderboard_compare.json"
echo "  - $WEBSITE_REPO/data/last_updated.json"
echo ""
echo "This script is an offline compatibility path."
echo "The website primary source of truth should remain the HF dataset snapshots."
