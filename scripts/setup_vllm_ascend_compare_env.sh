#!/usr/bin/env bash

set -euo pipefail

resolve_python_bin() {
    if [[ -n "${BENCH_VLLM_ASCEND_PY:-}" ]]; then
        printf '%s\n' "${BENCH_VLLM_ASCEND_PY}"
        return 0
    fi

    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ "${CONDA_DEFAULT_ENV:-}" != "base" ]]; then
        printf '%s/bin/python\n' "${CONDA_PREFIX}"
        return 0
    fi

    echo "ERROR: BENCH_VLLM_ASCEND_PY is not set and no non-base conda environment is active." >&2
    echo "Activate a target conda env first, or pass BENCH_VLLM_ASCEND_PY=/path/to/python." >&2
    exit 1
}

resolve_env_name() {
    if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        printf '%s\n' "${CONDA_DEFAULT_ENV}"
        return 0
    fi

    printf '%s\n' "$(basename "$(dirname "${PYTHON_BIN}")")"
}

detect_toolkit_home() {
    local candidates=()

    if [[ -n "${SAGELLM_ASCEND_TOOLKIT_HOME:-}" ]] && [[ -d "${SAGELLM_ASCEND_TOOLKIT_HOME}" ]]; then
        printf '%s\n' "${SAGELLM_ASCEND_TOOLKIT_HOME}"
        return 0
    fi

    candidates+=(
        "/usr/local/Ascend/ascend-toolkit/latest"
        "/usr/local/Ascend/ascend-toolkit.bak.8.1/latest"
    )

    while IFS= read -r candidate; do
        [[ -d "$candidate" ]] || continue
        candidates+=("$candidate")
    done < <(find /usr/local/Ascend -maxdepth 2 -mindepth 2 -type d -name toolkit 2>/dev/null | sort -u)

    local candidate
    for candidate in "${candidates[@]}"; do
        [[ -d "$candidate" ]] || continue
        if [[ -f "$candidate/version.info" ]] || [[ -d "$candidate/runtime" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

extract_cann_series() {
    local toolkit_home="$1"
    local version_file_candidates=(
        "${toolkit_home}/version.info"
        "${toolkit_home}/toolkit/version.info"
        "${toolkit_home}/runtime/version.info"
    )
    local version_dir=""
    local version_file

    for version_file in "${version_file_candidates[@]}"; do
        [[ -f "$version_file" ]] || continue
        version_dir="$(awk -F= '/^version_dir=/{print $2; exit}' "$version_file")"
        if [[ -n "$version_dir" ]]; then
            printf '%s\n' "$version_dir"
            return 0
        fi
    done

    printf '%s\n' "unknown"
}

select_profile() {
    local requested="${BENCH_ASCEND_PROFILE:-auto}"

    case "$requested" in
        official-v0.13.0|official-v0.11.0)
            printf '%s\n' "$requested"
            return 0
            ;;
        auto)
            ;;
        *)
            echo "ERROR: unknown BENCH_ASCEND_PROFILE=$requested (supported: auto, official-v0.11.0, official-v0.13.0)" >&2
            exit 1
            ;;
    esac

    case "$HOST_CANN_SERIES" in
        8.5*|8.5.*)
            printf '%s\n' "official-v0.13.0"
            ;;
        8.3*|8.3.*)
            printf '%s\n' "official-v0.11.0"
            ;;
        *)
            echo "ERROR: no official Ascend endpoint benchmark profile is configured for host CANN series '$HOST_CANN_SERIES'." >&2
            echo "Detected toolkit: ${TOOLKIT_HOME:-<missing>}" >&2
            echo "This host appears to use an older/non-target toolkit line; prefer an official vllm-ascend container or another dedicated machine/env matching the published matrix." >&2
            exit 1
            ;;
    esac
}

load_profile() {
    local profile="$1"

    case "$profile" in
        official-v0.11.0)
            BENCH_CANN_SERIES="8.3"
            BENCH_TORCH_VERSION="2.7.1"
            BENCH_TORCH_NPU_VERSION="2.7.1.post1"
            BENCH_TORCHVISION_VERSION="0.22.1"
            BENCH_TORCHAUDIO_VERSION="2.7.1"
            BENCH_TRANSFORMERS_VERSION="4.57.1"
            BENCH_VLLM_ASCEND_VERSION="0.11.0"
            BENCH_VLLM_VERSION="0.11.0"
            ;;
        official-v0.13.0)
            BENCH_CANN_SERIES="8.5"
            BENCH_TORCH_VERSION="2.8.0"
            BENCH_TORCH_NPU_VERSION="2.8.0.post2"
            BENCH_TORCHVISION_VERSION="0.23.0"
            BENCH_TORCHAUDIO_VERSION="2.8.0"
            BENCH_TRANSFORMERS_VERSION="4.57.4"
            BENCH_VLLM_ASCEND_VERSION="0.13.0"
            BENCH_VLLM_VERSION="0.13.0"
            ;;
        *)
            echo "ERROR: unsupported profile '$profile'" >&2
            exit 1
            ;;
    esac
}

ensure_host_matches_profile() {
    if [[ "${BENCH_SKIP_HOST_MATRIX_CHECK:-0}" == "1" ]]; then
        return 0
    fi

    if [[ "$HOST_CANN_SERIES" != ${BENCH_CANN_SERIES}* ]]; then
        echo "ERROR: selected profile '$ACTIVE_PROFILE' requires CANN ${BENCH_CANN_SERIES}.x, but host reports '$HOST_CANN_SERIES'." >&2
        echo "Detected toolkit: ${TOOLKIT_HOME:-<missing>}" >&2
        echo "Use a machine/container with the matching Ascend software stack, or override only if you are intentionally testing an unsupported combination." >&2
        exit 1
    fi
}

ensure_dedicated_env() {
    local env_name="$1"

    if [[ "${BENCH_VLLM_ASCEND_ALLOW_MAIN_ENV:-0}" == "1" ]]; then
        return 0
    fi

    if [[ "$env_name" == "sagellm" ]]; then
        echo "ERROR: refusing to prepare the Ascend compare stack inside the main 'sagellm' env." >&2
        echo "Use a dedicated non-base conda env, or export BENCH_VLLM_ASCEND_ALLOW_MAIN_ENV=1 to override." >&2
        exit 1
    fi
}

append_compare_packages() {
    COMPARE_PACKAGES=(
        "torch==${BENCH_TORCH_VERSION}"
        "torch-npu==${BENCH_TORCH_NPU_VERSION}"
        "torchvision==${BENCH_TORCHVISION_VERSION}"
        "torchaudio==${BENCH_TORCHAUDIO_VERSION}"
        "transformers==${BENCH_TRANSFORMERS_VERSION}"
        "vllm-ascend==${BENCH_VLLM_ASCEND_VERSION}"
    )

    if [[ "${BENCH_REQUIRE_VLLM:-1}" == "1" ]]; then
        COMPARE_PACKAGES+=("vllm==${BENCH_VLLM_VERSION}")
    fi
}

verify_compare_stack_resolves() {
    echo "[setup_vllm_ascend_compare_env] validating package resolver before mutating env"
    if "${PYTHON_BIN}" -m pip install --dry-run "${COMPARE_PACKAGES[@]}"; then
        return 0
    fi

    echo "ERROR: the requested Ascend compare stack is not resolver-compatible in this env." >&2
    echo "Requested versions:" >&2
    printf '  - %s\n' "${COMPARE_PACKAGES[@]}" >&2
    echo "" >&2
    echo "Official vLLM Ascend endpoint serving requires matching versions of vllm and vllm-ascend." >&2
    echo "On this host, the old local 0.11.0 + torch 2.7.1 plugin stack does not resolve with pip once full vllm is included." >&2
    echo "Recommended next step:" >&2
    echo "  1. Keep the main sagellm env unchanged." >&2
    echo "  2. Use a dedicated non-base conda env or the official vllm-ascend container." >&2
    echo "  3. Retry with an official same-version matrix via BENCH_VLLM_VERSION / BENCH_VLLM_ASCEND_VERSION and matching torch pins." >&2
    exit 1
}

PYTHON_BIN="$(resolve_python_bin)"
ENV_NAME="$(resolve_env_name)"
SAGELLM_ROOT="${SAGELLM_ROOT:-/home/shuhao/sagellm}"
ASCEND_WRAPPER="${SAGELLM_ROOT}/scripts/sagellm_with_ascend_env.sh"
TOOLKIT_HOME="$(detect_toolkit_home || true)"
HOST_CANN_SERIES="unknown"

if [[ -n "$TOOLKIT_HOME" ]]; then
    HOST_CANN_SERIES="$(extract_cann_series "$TOOLKIT_HOME")"
fi

ACTIVE_PROFILE="$(select_profile)"
load_profile "$ACTIVE_PROFILE"

BENCH_TORCH_VERSION="${BENCH_TORCH_VERSION:-$BENCH_TORCH_VERSION}"
BENCH_TORCH_NPU_VERSION="${BENCH_TORCH_NPU_VERSION:-$BENCH_TORCH_NPU_VERSION}"
BENCH_TORCHVISION_VERSION="${BENCH_TORCHVISION_VERSION:-$BENCH_TORCHVISION_VERSION}"
BENCH_TORCHAUDIO_VERSION="${BENCH_TORCHAUDIO_VERSION:-$BENCH_TORCHAUDIO_VERSION}"
BENCH_TRANSFORMERS_VERSION="${BENCH_TRANSFORMERS_VERSION:-$BENCH_TRANSFORMERS_VERSION}"
BENCH_VLLM_ASCEND_VERSION="${BENCH_VLLM_ASCEND_VERSION:-$BENCH_VLLM_ASCEND_VERSION}"
BENCH_VLLM_VERSION="${BENCH_VLLM_VERSION:-$BENCH_VLLM_VERSION}"

ensure_dedicated_env "$ENV_NAME"
ensure_host_matches_profile
append_compare_packages

echo "[setup_vllm_ascend_compare_env] python=${PYTHON_BIN}"
echo "[setup_vllm_ascend_compare_env] env=${ENV_NAME}"
echo "[setup_vllm_ascend_compare_env] sagellm_root=${SAGELLM_ROOT}"
echo "[setup_vllm_ascend_compare_env] profile=${ACTIVE_PROFILE}"
echo "[setup_vllm_ascend_compare_env] toolkit=${TOOLKIT_HOME:-<missing>}"
echo "[setup_vllm_ascend_compare_env] host_cann=${HOST_CANN_SERIES}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: python not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -x "${ASCEND_WRAPPER}" ]]; then
    echo "ERROR: Ascend wrapper not executable: ${ASCEND_WRAPPER}" >&2
    exit 1
fi

if [[ -z "${SAGELLM_ASCEND_TOOLKIT_HOME:-}" ]] && [[ -n "$TOOLKIT_HOME" ]]; then
    export SAGELLM_ASCEND_TOOLKIT_HOME="$TOOLKIT_HOME"
fi

verify_compare_stack_resolves

"${PYTHON_BIN}" -m pip install -U "${COMPARE_PACKAGES[@]}"

"${PYTHON_BIN}" -m pip check

"${ASCEND_WRAPPER}" "${PYTHON_BIN}" - <<'PY'
import torch, torch_npu

print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('npu_available', torch.npu.is_available())

if not torch.npu.is_available():
    raise RuntimeError('torch.npu.is_available() == False')

torch.npu.set_device('npu:0')
x = torch.ones(1, device='npu')
print('tensor_ok', (x + 1).cpu().tolist())
PY

echo "[setup_vllm_ascend_compare_env] OK"
