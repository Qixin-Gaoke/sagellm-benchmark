#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ACTION="${1:-start}"

IMAGE="${VLLM_ASCEND_IMAGE:-quay.io/ascend/vllm-ascend:v0.11.0-openeuler}"
CONTAINER_NAME="${VLLM_ASCEND_CONTAINER_NAME:-vllm-ascend-bench}"
MODEL="${VLLM_ASCEND_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_SOURCE="${VLLM_ASCEND_MODEL_SOURCE:-hf}"
HOST="${VLLM_ASCEND_HOST:-0.0.0.0}"
PORT="${VLLM_ASCEND_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${VLLM_ASCEND_TP_SIZE:-1}"
MAX_MODEL_LEN="${VLLM_ASCEND_MAX_MODEL_LEN:-4096}"
SHM_SIZE="${VLLM_ASCEND_SHM_SIZE:-16g}"
CACHE_DIR="${VLLM_ASCEND_CACHE_DIR:-$HOME/.cache/vllm-ascend-container}"
LOG_DIR="${VLLM_ASCEND_LOG_DIR:-$REPO_ROOT/benchmark_results/container_logs}"
LOG_FILE="$LOG_DIR/${CONTAINER_NAME}.log"
HEALTH_TIMEOUT_SEC="${VLLM_ASCEND_HEALTH_TIMEOUT_SEC:-900}"
DEVICE_LIST="${VLLM_ASCEND_DEVICES:-all}"
DOCKER_CMD="${DOCKER_CMD:-sudo -n docker}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"
PRIVILEGED_MODE="${VLLM_ASCEND_PRIVILEGED:-0}"
SECCOMP_UNCONFINED="${VLLM_ASCEND_SECCOMP_UNCONFINED:-1}"
CAP_SYS_NICE="${VLLM_ASCEND_CAP_SYS_NICE:-1}"

mkdir -p "$CACHE_DIR" "$LOG_DIR"

usage() {
    cat <<EOF
Usage: $0 [start|status|logs|stop|pull]

Environment variables:
  VLLM_ASCEND_IMAGE            Docker image tag
  VLLM_ASCEND_CONTAINER_NAME   Container name
  VLLM_ASCEND_MODEL            Model name
    VLLM_ASCEND_MODEL_SOURCE     hf or modelscope (default: hf)
  VLLM_ASCEND_PORT             Host port
  VLLM_ASCEND_TP_SIZE          Tensor parallel size
  VLLM_ASCEND_MAX_MODEL_LEN    Max model length
  VLLM_ASCEND_SHM_SIZE         Shared memory size
  VLLM_ASCEND_CACHE_DIR        Host cache directory mounted to /root/.cache
  VLLM_ASCEND_HEALTH_TIMEOUT_SEC  Readiness timeout
  VLLM_ASCEND_DEVICES          all or comma-separated davinci ids, e.g. 0,1
    VLLM_ASCEND_PRIVILEGED       1 to add --privileged
    VLLM_ASCEND_SECCOMP_UNCONFINED 1 to add --security-opt seccomp=unconfined
    VLLM_ASCEND_CAP_SYS_NICE     1 to add --cap-add SYS_NICE
  DOCKER_CMD                   Container runtime command (default: sudo -n docker)
EOF
}

docker_run() {
    $DOCKER_CMD "$@"
}

build_device_args() {
    local -a args=(
        --device /dev/davinci_manager
        --device /dev/devmm_svm
        --device /dev/hisi_hdc
    )
    local device_id
    local ids=()

    if [[ "$DEVICE_LIST" == "all" ]]; then
        while IFS= read -r device_id; do
            [[ -n "$device_id" ]] || continue
            ids+=("$device_id")
        done < <(find /dev -maxdepth 1 -type c -name 'davinci[0-9]*' -printf '%f\n' 2>/dev/null | sed 's/^davinci//' | sort -n)
    else
        IFS=',' read -r -a ids <<< "$DEVICE_LIST"
    fi

    for device_id in "${ids[@]}"; do
        device_id="${device_id//[[:space:]]/}"
        [[ -n "$device_id" ]] || continue
        args+=(--device "/dev/davinci${device_id}")
    done

    printf '%s\n' "${args[@]}"
}

container_exists() {
    docker_run ps -a --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" >/dev/null 2>&1
}

container_running() {
    docker_run ps --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" >/dev/null 2>&1
}

wait_ready() {
    local deadline=$((SECONDS + HEALTH_TIMEOUT_SEC))
    local url="http://127.0.0.1:${PORT}/v1/models"

    while (( SECONDS < deadline )); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "[run_vllm_ascend_container] ready: $url"
            return 0
        fi
        sleep 5
    done

    echo "ERROR: container did not become ready within ${HEALTH_TIMEOUT_SEC}s" >&2
    echo "See logs: $LOG_FILE" >&2
    return 1
}

pull_image() {
    echo "[run_vllm_ascend_container] pulling image $IMAGE"
    docker_run pull "$IMAGE"
}

start_container() {
    local -a device_args=()
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        device_args+=("$line")
    done < <(build_device_args)

    if container_exists; then
        echo "[run_vllm_ascend_container] removing existing container $CONTAINER_NAME"
        docker_run rm -f "$CONTAINER_NAME" >/dev/null
    fi

    pull_image

    local -a env_args=(
        -e "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
        -e "VLLM_WORKER_MULTIPROC_METHOD=spawn"
        -e "HF_ENDPOINT=${HF_ENDPOINT_VALUE}"
        -e "OPENBLAS_NUM_THREADS=1"
        -e "OMP_NUM_THREADS=1"
        -e "MKL_NUM_THREADS=1"
        -e "NUMEXPR_MAX_THREADS=1"
        -e "TOKENIZERS_PARALLELISM=false"
    )

    case "$MODEL_SOURCE" in
        hf)
            env_args+=( -e "VLLM_USE_MODELSCOPE=false" )
            ;;
        modelscope)
            env_args+=( -e "VLLM_USE_MODELSCOPE=true" )
            ;;
        *)
            echo "ERROR: unsupported VLLM_ASCEND_MODEL_SOURCE=$MODEL_SOURCE (expected hf or modelscope)" >&2
            exit 1
            ;;
    esac

    if [[ -n "$HF_TOKEN_VALUE" ]]; then
        env_args+=( -e "HF_TOKEN=${HF_TOKEN_VALUE}" )
    fi

    echo "[run_vllm_ascend_container] starting $CONTAINER_NAME on port $PORT"
    local -a security_args=()

    if [[ "$SECCOMP_UNCONFINED" == "1" ]]; then
        security_args+=(--security-opt seccomp=unconfined)
    fi
    if [[ "$CAP_SYS_NICE" == "1" ]]; then
        security_args+=(--cap-add SYS_NICE)
    fi
    if [[ "$PRIVILEGED_MODE" == "1" ]]; then
        security_args+=(--privileged)
    fi

    docker_run run -d \
        --name "$CONTAINER_NAME" \
        --network host \
        --shm-size "$SHM_SIZE" \
        --pids-limit -1 \
        "${security_args[@]}" \
        "${device_args[@]}" \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v "$CACHE_DIR:/root/.cache" \
        "${env_args[@]}" \
        "$IMAGE" \
        vllm serve "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager \
        > "$LOG_FILE"

    wait_ready
}

status_container() {
    if ! container_exists; then
        echo "[run_vllm_ascend_container] container not found: $CONTAINER_NAME"
        return 1
    fi

    docker_run ps -a --filter "name=^/${CONTAINER_NAME}$"
    curl -fsS "http://127.0.0.1:${PORT}/v1/models" || true
}

logs_container() {
    if container_exists; then
        docker_run logs --tail 200 "$CONTAINER_NAME"
        return 0
    fi

    if [[ -f "$LOG_FILE" ]]; then
        tail -200 "$LOG_FILE"
        return 0
    fi

    echo "[run_vllm_ascend_container] no logs found"
}

stop_container() {
    if ! container_exists; then
        echo "[run_vllm_ascend_container] container not found: $CONTAINER_NAME"
        return 0
    fi
    docker_run rm -f "$CONTAINER_NAME"
}

case "$ACTION" in
    start)
        start_container
        ;;
    status)
        status_container
        ;;
    logs)
        logs_container
        ;;
    stop)
        stop_container
        ;;
    pull)
        pull_image
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
