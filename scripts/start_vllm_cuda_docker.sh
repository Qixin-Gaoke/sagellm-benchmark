#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Start or reuse a dedicated vLLM CUDA Docker container for benchmark comparisons.

Environment variables:
  VLLM_CONTAINER_NAME            Container name (default: sagellm-benchmark-vllm)
  VLLM_DOCKER_IMAGE              Docker image (default: vllm/vllm-openai:v0.7.3)
  VLLM_MODEL                     Model to serve (default: Qwen/Qwen2.5-1.5B-Instruct)
  VLLM_PORT                      Host port mapped to container 8000 (default: 8000)
  VLLM_HOST                      Host bind address (default: 127.0.0.1)
    VLLM_DOCKER_NETWORK_MODE       Docker network mode: host or bridge (default: host)
  VLLM_GPU_DEVICE                GPU device id for --gpus device=... (default: 1)
  VLLM_DTYPE                     vLLM dtype (default: float16)
  VLLM_GPU_MEMORY_UTILIZATION    GPU memory utilization (default: 0.85)
  VLLM_MAX_MODEL_LEN             Max model length (default: 4096)
  VLLM_SHM_SIZE                  Container shared memory size (default: 16g)
  VLLM_READY_TIMEOUT             Seconds to wait for /v1/models (default: 600)
  VLLM_HF_CACHE_DIR              Host HF cache mount (default: ~/.cache/huggingface)
  VLLM_API_KEY                   Optional OpenAI API key for the server
    VLLM_DOCKER_AUTO_REMOVE        Set to 1 to use --rm for the container (default: 0)
  VLLM_EXTRA_ARGS                Extra args appended to the vLLM server command

Examples:
  VLLM_GPU_DEVICE=1 VLLM_PORT=9100 ./scripts/start_vllm_cuda_docker.sh
  VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct ./scripts/start_vllm_cuda_docker.sh
EOF
    exit 0
fi

CONTAINER_NAME="${VLLM_CONTAINER_NAME:-sagellm-benchmark-vllm}"
IMAGE="${VLLM_DOCKER_IMAGE:-vllm/vllm-openai:v0.7.3}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
NETWORK_MODE="${VLLM_DOCKER_NETWORK_MODE:-host}"
GPU_DEVICE="${VLLM_GPU_DEVICE:-1}"
DTYPE="${VLLM_DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
SHM_SIZE="${VLLM_SHM_SIZE:-16g}"
READY_TIMEOUT="${VLLM_READY_TIMEOUT:-600}"
HF_CACHE_DIR="${VLLM_HF_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}"
API_KEY="${VLLM_API_KEY:-${OPENAI_API_KEY:-}}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
AUTO_REMOVE="${VLLM_DOCKER_AUTO_REMOVE:-0}"
READY_URL="http://${HOST}:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || {
    echo "docker is required" >&2
    exit 1
}

docker info --format '{{json .Runtimes}}' | grep -q 'nvidia' || {
    echo "docker nvidia runtime is not configured" >&2
    exit 1
}

mkdir -p "$HF_CACHE_DIR"

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    if curl -fsS --max-time 5 "$READY_URL" >/dev/null 2>&1; then
        echo "vLLM container '$CONTAINER_NAME' is already ready at $READY_URL"
        exit 0
    fi

    echo "Existing container '$CONTAINER_NAME' is running but not ready; recreating it"
    docker rm -f "$CONTAINER_NAME" >/dev/null
elif docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "Removing exited container '$CONTAINER_NAME'"
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker_args=(
    run
    -d
    --name "$CONTAINER_NAME"
    --runtime nvidia
    --gpus "device=${GPU_DEVICE}"
    --ipc=host
    --shm-size "$SHM_SIZE"
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
)

if [[ "$NETWORK_MODE" == "host" ]]; then
    docker_args+=( --network host )
else
    docker_args+=( -p "${HOST}:${PORT}:8000" )
fi

if [[ "$AUTO_REMOVE" == "1" ]]; then
    docker_args+=( --rm )
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    docker_args+=( -e "HF_TOKEN=${HF_TOKEN}" )
fi

if [[ -n "$API_KEY" ]]; then
    docker_args+=( -e "VLLM_API_KEY=${API_KEY}" )
fi

server_args=(
    --model "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
)

if [[ -n "$API_KEY" ]]; then
    server_args+=( --api-key "$API_KEY" )
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args=( $EXTRA_ARGS )
    server_args+=( "${extra_args[@]}" )
fi

echo "Starting vLLM Docker container '$CONTAINER_NAME' from image '$IMAGE'"
echo "  model=$MODEL"
echo "  endpoint=http://${HOST}:${PORT}/v1"
echo "  gpu=$GPU_DEVICE"

docker "${docker_args[@]}" "$IMAGE" "${server_args[@]}" >/dev/null

deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS --max-time 5 "$READY_URL" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for vLLM Docker endpoint at $READY_URL" >&2
        docker logs "$CONTAINER_NAME" | tail -n 200 >&2 || true
        exit 1
    fi
    sleep 5
done

echo "vLLM Docker endpoint is ready at $READY_URL"