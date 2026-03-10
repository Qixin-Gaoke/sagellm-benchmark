#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Stop the dedicated vLLM CUDA Docker container used for benchmark comparisons.

Environment variables:
  VLLM_CONTAINER_NAME   Container name (default: sagellm-benchmark-vllm)
EOF
    exit 0
fi

CONTAINER_NAME="${VLLM_CONTAINER_NAME:-sagellm-benchmark-vllm}"

command -v docker >/dev/null 2>&1 || {
    echo "docker is required" >&2
    exit 1
}

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME"
    echo "Stopped $CONTAINER_NAME"
else
    echo "Container '$CONTAINER_NAME' does not exist"
fi