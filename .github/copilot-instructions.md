# GitHub Copilot Instructions

## Version Source of Truth (Mandatory)

For Python packages in this repository, version must have exactly one hardcoded source:

1. Only one hardcoded version location is allowed: `src/<package>/_version.py`
2. `pyproject.toml` must use dynamic version:
   - `[project] dynamic = ["version"]`
   - `[tool.setuptools.dynamic] version = {attr = "<package>._version.__version__"}`
3. `src/<package>/__init__.py` must import version from `_version.py`:
   - `from <package>._version import __version__`
4. Do not hardcode version in `pyproject.toml` (`project.version`) or `__init__.py`
5. For version bump, update only `_version.py`

## Reminder

When asked to update package version, change only `_version.py`.

## Ascend Endpoint Benchmarking Reminder (Mandatory)

- `sagellm-benchmark` is the owning repo for third-party engine comparison workflows. Keep `vLLM` / `LMDeploy` / other compare-engine dependencies, install helpers, endpoint probes, and live benchmark orchestration here rather than in `sagellm-core`.
- `sagellm-benchmark` owns third-party engine comparison responsibilities: dependency extras, convenience install scripts, endpoint liveness checks, and live metrics collection all stay on the benchmark side.
- For the standard `sageLLM vs vLLM` workflow, prefer the dedicated benchmark CLI:
   - `sagellm-benchmark vllm-compare install-ascend`
   - `sagellm-benchmark vllm-compare run --sagellm-url <url> --vllm-url <url> --model <model>`
- `sagellm-benchmark compare` remains the generic multi-endpoint entrypoint; `vllm-compare` is the thin, semantic wrapper for the common vLLM comparison path.
- `scripts/setup_vllm_ascend_compare_env.sh` is a compatibility/ops layer only. Do not add new primary logic there when the same behavior belongs in the CLI.
- Benchmark dependency declarations must live in `pyproject.toml` extras. Scripts may pin a validated environment matrix, but must not become an alternate source of truth for compare-client dependencies.

- In Ascend environments with only `vllm-ascend` installed, `python -m vllm.entrypoints.openai.api_server` may fail because `vllm` package/module is absent.
- Before assuming an endpoint is `vllm-ascend`, always verify with runtime checks:
   - `ss -ltnp | grep -E ':<port>'` to map port → process
   - `curl /health`, `curl /info`, `curl /v1/models` to identify engine capability/shape
- Do not label a benchmark as `vllm-ascend` unless the serving process is confirmed to be a real `vllm-ascend` server process.

### train05 / current Ascend host practical rules

- Prefer `sagellm-benchmark vllm-compare install-ascend` over directly invoking setup shell scripts in new instructions, examples, or automation.
- Prefer `sagellm-benchmark vllm-compare run` over passing anonymous `endpoint_a/endpoint_b` style arguments in new instructions, examples, or automation.
- Always inject Ascend runtime before startup via wrapper:
   - `cd /home/user8/sagellm`
   - `./scripts/sagellm_with_ascend_env.sh <python-or-server-command>`
- Run preflight smoke test in the exact target env before server startup:
   - `import torch, torch_npu`
   - `torch.npu.is_available()`
   - `torch.npu.set_device('npu:0')` and one small NPU tensor op
- If `python -m vllm.entrypoints.openai.api_server` fails with module-missing/entrypoint-mismatch, treat it as environment/package-layout issue first; do not continue benchmark with an unverified server.
- Startup success criteria are mandatory (all pass):
   - Port bind: `ss -ltnp | grep -E ':<port>'`
   - Process fingerprint: `pgrep -af 'vllm|vllm-ascend|EngineCore'`
   - API health/model endpoints return valid payloads
- If any criterion fails, mark run as invalid and stop comparison output generation.


## Git Hooks（强制 - Mandatory）

🚨 **所有开发者必须安装 pre-commit 和 pre-push hooks，绝对不允许跳过。**

### 安装要求

1. 克隆仓库后，**第一件事**必须运行 `./quickstart.sh` 安装 Git hooks
2. 如果仓库提供 `hooks/` 目录，必须确保 `.git/hooks/pre-commit` 和 `.git/hooks/pre-push` 已正确链接或复制
3. 每次 `git commit` 和 `git push` 都必须经过 hooks 检查（ruff format / ruff check / pytest 等）

### 禁止绕过 Hooks

- ❌ **禁止** 使用 `git commit --no-verify` 或 `git push --no-verify` 跳过 hooks
- ❌ **禁止** 删除、禁用或修改 `.git/hooks/pre-commit` / `.git/hooks/pre-push`
- ❌ **禁止** 通过任何方式（环境变量、配置等）绕过 hooks 检查
- ⚠️ `--no-verify` 仅在极端特殊情况下允许（如修复 CI 基础设施本身），且必须在 commit message 中注明原因

### Copilot Agent 行为规范

- ✅ 执行 `git commit` 或 `git push` 时，**永远不要**添加 `--no-verify` 标志
- ✅ 如果 hooks 检查失败，必须先修复问题再提交，而不是绕过 hooks
- ✅ 帮助开发者设置 hooks 时，推荐运行 `./quickstart.sh`
- ✅ 默认 `git push` **不会自动发布**，发布必须显式触发
- ✅ 如需在推送 `main-dev` 时发布，使用 `git push -o sagellm-publish origin main-dev`；若当前 Git 客户端不支持 push option，则使用 `SAGELLM_PUBLISH_ON_PUSH=1 git push origin main-dev`

## 🚫 NEVER_CREATE_DOT_VENV_MANDATORY

- 永远不要创建 `.venv` 或 `venv`（无任何例外）。
- NEVER create `.venv`/`venv` in this repository under any circumstance.
- 必须复用当前已配置的非-venv Python 环境（如现有 conda 环境）。
- If any script/task suggests creating a virtualenv, skip that step and continue with the existing environment.
