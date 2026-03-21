#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Stop the dedicated vLLM CUDA Docker container used for benchmark comparisons.

Environment variables:
    DOCKER_CMD             Container runtime command (default: docker)
  VLLM_CONTAINER_NAME   Container name (default: sagellm-benchmark-vllm)
EOF
    exit 0
fi

CONTAINER_NAME="${VLLM_CONTAINER_NAME:-sagellm-benchmark-vllm}"
DOCKER_CMD="${DOCKER_CMD:-docker}"

read -r -a docker_cmd <<< "$DOCKER_CMD"

docker_run() {
    "${docker_cmd[@]}" "$@"
}

command -v "${docker_cmd[0]}" >/dev/null 2>&1 || {
    echo "${docker_cmd[0]} is required" >&2
    exit 1
}

if docker_run ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    docker_run rm -f "$CONTAINER_NAME"
    echo "Stopped $CONTAINER_NAME"
else
    echo "Container '$CONTAINER_NAME' does not exist"
fi
