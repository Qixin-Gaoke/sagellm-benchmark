#!/usr/bin/env bash
# quickstart.sh — sagellm-benchmark environment setup
#
# 安装语义：
# - standard: 依赖优先从 PyPI 安装（稳定/发布导向）
# - dev:      在 standard 基础上，尽量用本地 editable 覆盖（--no-deps）

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PARENT_DIR="$(dirname "$PROJECT_ROOT")"

INSTALL_MODE="dev"
DOCTOR="false"
SKIP_HOOKS="false"
SKIP_CLEANUP="false"
YES="false"

show_help() {
    echo "sagellm-benchmark Quick Start"
    echo ""
    echo "用法:"
    echo "  ./quickstart.sh                 默认开发安装 (--dev)"
    echo "  ./quickstart.sh --dev           开发模式：先安装 PyPI 基线，再用本地 editable 覆盖"
    echo "  ./quickstart.sh --standard      标准模式：依赖从 PyPI 安装，当前仓库 editable 安装"
    echo "  ./quickstart.sh --doctor        仅做环境诊断"
    echo "  ./quickstart.sh --skip-cleanup  跳过 isagellm-* 历史包清理"
    echo "  ./quickstart.sh --skip-hooks    跳过 Git hooks 安装"
    echo "  ./quickstart.sh --yes|-y        非交互模式（保留兼容参数）"
    echo "  ./quickstart.sh --help          显示帮助"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dev)
                INSTALL_MODE="dev"
                ;;
            --standard)
                INSTALL_MODE="standard"
                ;;
            --doctor)
                DOCTOR="true"
                ;;
            --skip-hooks)
                SKIP_HOOKS="true"
                ;;
            --skip-cleanup)
                SKIP_CLEANUP="true"
                ;;
            --yes|-y)
                YES="true"
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}❌ 未知参数: $1${NC}"
                echo ""
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

detect_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}❌ 未找到可用 Python 命令（python3/python）${NC}"
        exit 1
    fi
    PIP_CMD=("$PYTHON_CMD" -m pip)
}

run_with_diagnostics() {
    local label="$1"
    shift
    local log_file
    log_file=$(mktemp)

    if "$@" >"$log_file" 2>&1; then
        rm -f "$log_file"
        return 0
    fi

    echo -e "${RED}❌ ${label} 失败${NC}"
    echo -e "${YELLOW}--- 详细错误日志开始 ---${NC}"
    cat "$log_file"
    echo -e "${YELLOW}--- 详细错误日志结束 ---${NC}"
    rm -f "$log_file"
    return 1
}

run_doctor() {
    echo -e "${BOLD}${BLUE}Environment Diagnosis${NC}"
    echo ""
    echo -e "${YELLOW}Python:${NC} $($PYTHON_CMD --version 2>/dev/null || echo 'NOT FOUND')"
    echo -e "${YELLOW}Conda env:${NC} ${CONDA_DEFAULT_ENV:-none}"
    echo -e "${YELLOW}Venv:${NC} ${VIRTUAL_ENV:-none}"
    echo -e "${YELLOW}ruff:${NC} $(ruff --version 2>/dev/null || echo 'NOT FOUND')"
    echo -e "${YELLOW}pytest:${NC} $(pytest --version 2>/dev/null || echo 'NOT FOUND')"
    echo ""
    echo -e "${YELLOW}Git hooks installed:${NC}"
    for h in pre-commit pre-push post-commit; do
        if [ -f "$PROJECT_ROOT/.git/hooks/$h" ]; then
            echo -e "  ${GREEN}✓ $h${NC}"
        else
            echo -e "  ${RED}✗ $h${NC}"
        fi
    done
}

check_environment() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        echo -e "${RED}❌ Detected Python venv: ${VIRTUAL_ENV}${NC}"
        echo -e "${YELLOW}👉 This repository forbids venv/.venv. Please use an existing non-venv env.${NC}"
        exit 1
    fi

    local py_version
    py_version="$($PYTHON_CMD --version | awk '{print $2}')"
    if ! $PYTHON_CMD -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
        echo -e "${RED}❌ Python ${py_version} 版本过低，需要 >= 3.11${NC}"
        exit 1
    fi

    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        echo -e "${GREEN}✓ Conda env: ${CONDA_DEFAULT_ENV}${NC}"
    else
        echo -e "${YELLOW}⚠ 未检测到 Conda 环境，请确认当前为已配置的非-venv Python 环境${NC}"
    fi
    echo -e "${GREEN}✓ Python ${py_version}${NC}"
}

cleanup_isagellm_packages() {
    mapfile -t installed_packages < <(
        "$PYTHON_CMD" - <<'PY'
import importlib.metadata as metadata

names = set()
for dist in metadata.distributions():
    name = dist.metadata.get("Name")
    if not name:
        continue
    lowered = name.strip().lower()
    if lowered.startswith("isagellm-"):
        names.add(lowered)

for item in sorted(names):
    print(item)
PY
    )

    if [ "${#installed_packages[@]}" -eq 0 ]; then
        echo -e "${GREEN}✓ 无需清理（未发现 isagellm-* 已安装包）${NC}"
        return 0
    fi

    echo -e "${BLUE}🧹 发现 ${#installed_packages[@]} 个 isagellm-* 包，开始清理...${NC}"
    local pkg
    for pkg in "${installed_packages[@]}"; do
        echo -e "  ${CYAN}- uninstall ${pkg}${NC}"
        run_with_diagnostics "卸载 ${pkg}" "${PIP_CMD[@]}" uninstall -y "$pkg"
    done
    echo -e "${GREEN}✓ isagellm-* 包清理完成${NC}"
}

install_pypi_baseline() {
    local deps=(
        "isagellm-protocol"
        "isagellm-core"
        "isagellm-backend"
    )
    echo -e "${BLUE}📦 从 PyPI 安装基础依赖: ${deps[*]}${NC}"
    run_with_diagnostics "安装 PyPI 基线依赖" "${PIP_CMD[@]}" install "${deps[@]}"
    echo -e "${GREEN}✓ PyPI 基线依赖安装完成${NC}"
}

install_current_repo() {
    if [ "$INSTALL_MODE" = "dev" ]; then
        local dev_deps=(
            "pytest>=7.0.0"
            "pytest-cov>=4.0.0"
            "pytest-asyncio>=0.21.0"
            "ruff>=0.1.0"
            "httpx>=0.24.0"
            "isage-pypi-publisher>=0.2.0.0"
        )

        echo -e "${BLUE}📦 安装开发工具依赖（避免 .[dev] 自引用回溯）${NC}"
        run_with_diagnostics "安装开发工具依赖" "${PIP_CMD[@]}" install "${dev_deps[@]}"

        echo -e "${BLUE}📦 安装当前仓库（editable, --no-deps）${NC}"
        run_with_diagnostics "安装当前仓库 . (editable --no-deps)" "${PIP_CMD[@]}" install -e . --no-deps
    else
        echo -e "${BLUE}📦 安装当前仓库（editable）${NC}"
        run_with_diagnostics "安装当前仓库 ." "${PIP_CMD[@]}" install -e .
    fi
    echo -e "${GREEN}✓ 当前仓库安装完成${NC}"
}

install_local_editable_overrides() {
    local repos=(
        "sagellm-protocol"
        "sagellm-core"
        "sagellm-backend"
    )

    local repo_path
    for repo in "${repos[@]}"; do
        repo_path="$PARENT_DIR/$repo"
        if [ -f "$repo_path/pyproject.toml" ]; then
            echo -e "${BLUE}🔁 本地覆盖: ${repo_path} (editable, --no-deps)${NC}"
            run_with_diagnostics "本地覆盖 $repo" "${PIP_CMD[@]}" install -e "$repo_path" --no-deps
            echo -e "${GREEN}✓ 已使用本地仓库覆盖: $repo${NC}"
        else
            echo -e "${YELLOW}⚠️ 未找到本地仓库 $repo，保留 PyPI 版本${NC}"
        fi
    done
}

ascend_hw_detected() {
    if ! command -v npu-smi >/dev/null 2>&1; then
        return 1
    fi

    local npu_info
    npu_info="$(npu-smi info 2>/dev/null || true)"
    [[ -n "$npu_info" && "$npu_info" == *"NPU"* ]]
}

install_benchmark_extra() {
    local extra_name="$1"
    echo -e "${BLUE}📦 安装 benchmark extra: ${extra_name}${NC}"
    run_with_diagnostics \
        "安装 benchmark extra ${extra_name}" \
        "${PIP_CMD[@]}" install -e ".[$extra_name]"
    echo -e "${GREEN}✓ 已安装 benchmark extra: ${extra_name}${NC}"
}

install_optional_vllm_client() {
    if ascend_hw_detected; then
        echo -e "${BLUE}📦 检测到 Ascend，先安装 canonical extra，再叠加已验证版本矩阵${NC}"
        install_benchmark_extra "vllm-ascend-client"
        run_with_diagnostics "卸载 vllm" "${PIP_CMD[@]}" uninstall -y vllm || true
        run_with_diagnostics "安装 vllm-ascend" "${PIP_CMD[@]}" install "vllm-ascend>=0.11.0"
        echo -e "${GREEN}✓ 已完成 Ascend compare client 安装（extras + 便利层矩阵）${NC}"
    else
        echo -e "${BLUE}📦 未检测到 Ascend，安装 canonical vLLM compare extra${NC}"
        install_benchmark_extra "vllm-client"
        echo -e "${GREEN}✓ 已完成通用 vLLM compare client 安装（以 extras 为准）${NC}"
    fi
}

install_hooks() {
    if [ ! -d "$PROJECT_ROOT/.git/hooks" ]; then
        echo -e "${YELLOW}⚠ .git/hooks 目录不存在，跳过 hooks 安装${NC}"
        return 0
    fi

    if [ -d "$PROJECT_ROOT/hooks" ]; then
        local installed=0
        local hook_src
        for hook_src in "$PROJECT_ROOT/hooks"/*; do
            local hook_name
            hook_name=$(basename "$hook_src")
            local hook_dst
            hook_dst="$PROJECT_ROOT/.git/hooks/$hook_name"
            if [ "$(realpath "$hook_src")" = "$(realpath "$hook_dst")" ]; then
                chmod +x "$hook_dst"
                echo -e "  ${GREEN}✓ $hook_name (already installed)${NC}"
                installed=$((installed + 1))
                continue
            fi
            cp "$hook_src" "$hook_dst"
            chmod +x "$hook_dst"
            echo -e "  ${GREEN}✓ $hook_name${NC}"
            installed=$((installed + 1))
        done
        echo -e "${GREEN}✓ $installed hook(s) installed${NC}"
    else
        echo -e "${YELLOW}⚠ hooks/ directory not found — skipping${NC}"
    fi
}

main() {
    parse_args "$@"
    detect_python

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  sagellm-benchmark — Quick Start${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Install mode:${NC} ${INSTALL_MODE}"
    echo ""

    if [ "$DOCTOR" = "true" ]; then
        run_doctor
        exit 0
    fi

    echo -e "${YELLOW}${BOLD}Step 1/6: Checking Python environment${NC}"
    check_environment
    echo ""

    echo -e "${YELLOW}${BOLD}Step 2/6: Cleaning existing isagellm-* packages${NC}"
    if [ "$SKIP_CLEANUP" = "true" ]; then
        echo -e "${YELLOW}⚠ 已跳过清理（--skip-cleanup）${NC}"
    else
        cleanup_isagellm_packages
    fi
    echo ""

    echo -e "${YELLOW}${BOLD}Step 3/6: Installing PyPI baseline${NC}"
    install_pypi_baseline
    echo ""

    echo -e "${YELLOW}${BOLD}Step 4/6: Installing editable package(s)${NC}"
    install_current_repo
    if [ "$INSTALL_MODE" = "dev" ]; then
        install_local_editable_overrides
    fi
    echo ""

    echo -e "${YELLOW}${BOLD}Step 5/6: Installing optional compare client${NC}"
    install_optional_vllm_client
    echo ""

    echo -e "${YELLOW}${BOLD}Step 6/6: Installing Git hooks${NC}"
    if [ "$SKIP_HOOKS" = "true" ]; then
        echo -e "${YELLOW}⚠ 已跳过 hooks 安装（--skip-hooks）${NC}"
    else
        install_hooks
    fi
    echo ""

    echo -e "${GREEN}${BOLD}✓ Setup complete!${NC}"
    echo ""
    echo -e "${YELLOW}Dependency note:${NC} benchmark extras remain the source of truth; quickstart only installs the matching extra and optional validated pins."
    echo ""
    echo -e "${BLUE}${BOLD}Next steps:${NC}"
    echo -e "  ${CYAN}pytest tests/${NC}                    — run tests"
    echo -e "  ${CYAN}ruff check src/${NC}                  — lint"
    echo -e "  ${CYAN}./quickstart.sh --standard${NC}       — install PyPI baseline + local editable"
    echo -e "  ${CYAN}./quickstart.sh --dev${NC}            — standard + local editable overrides"
    echo -e "  ${CYAN}./quickstart.sh --doctor${NC}         — diagnose environment"
    echo ""
}

main "$@"
