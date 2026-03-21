# Ascend Endpoint Benchmark Guide

本文档记录在 Ascend 机器上准备 `vllm-ascend` 与 `sagellm` endpoint 对比时，当前仓库采用的安全策略与最小可执行流程。

目标：后续继续做性能比较时，不需要再重新摸索环境安装、启动命令与判活步骤。

## 1. 适用范围

- 机器具备 Ascend NPU（`npu-smi` 可用）
- 使用 OpenAI-compatible endpoint 做 live benchmark
- 对比对象：
  - `vllm-ascend`
  - `sagellm serve --benchmark-mode`

## 2. 推荐策略

当前仓库默认采用下面这条策略，而不是把完整 `vllm` 栈直接塞进 umbrella 主环境：

1. `sagellm` 主环境保持稳定，不自动安装完整 `vllm + vllm-ascend` endpoint 栈。
2. `vllm-ascend` 对比侧必须使用“同版本 `vllm` + `vllm-ascend`”的官方矩阵。
3. 对比环境必须与主 `sagellm` 环境隔离，推荐使用单独的非 `base` conda 环境，或官方 `vllm-ascend` 容器。

原因：官方文档和 FAQ 都要求 `vllm-ascend` 与 `vllm` 保持同版本；而当前机器上旧的本地 `0.11.0 + torch 2.7.1` 插件栈在加入完整 `vllm` 后会出现 pip resolver 冲突，不能再假装这是“一键可安装的 endpoint compare 环境”。

当前机器上已观测到的冲突点：

- 本地旧栈：`torch==2.7.1`、`torch-npu==2.7.1`、`vllm-ascend==0.11.0`
- `pip install --dry-run vllm==0.11.0 ...` 解析结果要求 `vllm==0.11.0` 依赖 `torch==2.8.0`
- 因此，当前旧栈不能直接扩展成完整 `vllm` endpoint benchmark 环境

## 3. 官方 profile

当前脚本支持的官方 profile：

- `official-v0.11.0`
  - 目标 CANN: `8.3.x`
  - 目标栈: `vllm==0.11.0`, `vllm-ascend==0.11.0`
- `official-v0.13.0`
  - 目标 CANN: `8.5.x`
  - 目标栈: `vllm==0.13.0`, `vllm-ascend==0.13.0`

默认 `BENCH_ASCEND_PROFILE=auto`，脚本会按主机 CANN 版本自动选 profile：

- `8.5.x` -> `official-v0.13.0`
- `8.3.x` -> `official-v0.11.0`
- 其他版本 -> 直接 fail-fast，并提示改用匹配机器或官方容器

对于当前这类 `8.1.RC1` 主机，优先建议直接使用容器化路径，而不是继续尝试本机 Python 环境拼装完整 endpoint 栈。

## 4. 容器化启动 `vllm-ascend`（推荐于旧 CANN 主机）

仓库内提供了容器启动脚本：

```bash
cd sagellm-benchmark
bash scripts/run_vllm_ascend_container.sh start
```

默认行为：

- 使用 `quay.io/ascend/vllm-ascend:v0.11.0-openeuler`
- 使用 `sudo -n docker`
- 映射 `/dev/davinci*`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`
- 使用 host network，默认监听 `127.0.0.1:8000`
- 挂载本机模型缓存到容器内 `/root/.cache`
- 启动后自动轮询 `/v1/models` 做 ready check

常用操作：

```bash
bash scripts/run_vllm_ascend_container.sh status
bash scripts/run_vllm_ascend_container.sh logs
bash scripts/run_vllm_ascend_container.sh stop
```

常用参数：

```bash
VLLM_ASCEND_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
VLLM_ASCEND_PORT=8000 \
VLLM_ASCEND_TP_SIZE=1 \
bash scripts/run_vllm_ascend_container.sh start
```

如果只想使用部分卡：

```bash
VLLM_ASCEND_DEVICES=0,1 bash scripts/run_vllm_ascend_container.sh start
```

## 5. 一键安装对比环境

推荐直接使用 CLI：

```bash
sagellm-benchmark vllm-compare install-ascend
```

兼容脚本 `scripts/setup_vllm_ascend_compare_env.sh` 仍保留，但只是对上述 CLI 的薄包装。

默认行为：

- 优先使用 `BENCH_VLLM_ASCEND_PY`
- 若未显式传入，则使用当前已激活的非 `base` conda 环境对应的 `python`
- 默认按 `BENCH_ASCEND_PROFILE` 选择官方矩阵
- 若主机 CANN 版本与 profile 不匹配，会在安装前直接失败
- 默认要求完整 endpoint compare 栈包含 `vllm` 与 `vllm-ascend`
- 在真正安装前先执行 `pip install --dry-run` 做 resolver 校验
- 若当前版本矩阵不可解，则 fail-fast 退出，不污染当前环境
- 仅在 dry-run 可解时才继续安装并执行最小 Ascend 烟测（`torch + torch_npu + npu tensor`）

重要约束：

- 脚本默认拒绝在主 `sagellm` conda 环境中执行
- 如确需覆盖该保护，必须显式传入 `BENCH_VLLM_ASCEND_ALLOW_MAIN_ENV=1`

若机器的 Ascend toolkit 不在标准 `/usr/local/Ascend/ascend-toolkit` 布局下，需显式传入：

```bash
SAGELLM_ASCEND_TOOLKIT_HOME=/actual/ascend/toolkit/path \
bash scripts/setup_vllm_ascend_compare_env.sh
```

如需覆盖 Python 路径：

```bash
BENCH_VLLM_ASCEND_PY=/path/to/python sagellm-benchmark vllm-compare install-ascend
```

如需显式指定官方 profile：

```bash
BENCH_ASCEND_PROFILE=official-v0.13.0 bash scripts/setup_vllm_ascend_compare_env.sh
```

如需覆盖 profile 内的单项版本，可显式传入：

```bash
BENCH_VLLM_VERSION=0.13.0 \
BENCH_VLLM_ASCEND_VERSION=0.13.0 \
BENCH_TORCH_VERSION=2.8.0 \
BENCH_TORCH_NPU_VERSION=2.8.0.post2 \
bash scripts/setup_vllm_ascend_compare_env.sh
```

前提是当前机器的 CANN / torch-npu / Python 版本也满足该官方矩阵。

## 6. 启动前烟测

在任意 `vllm-ascend` 启动前，必须先注入 Ascend 运行时环境：

```bash
cd /home/user8/sagellm
./scripts/sagellm_with_ascend_env.sh python - <<'PY'
import torch, torch_npu
print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('npu_available', torch.npu.is_available())
torch.npu.set_device('npu:0')
x = torch.ones(1, device='npu')
print('tensor_ok', (x + 1).cpu().tolist())
PY
```

任一步失败都不应继续 benchmark。

## 7. 启动 `vllm-ascend`

若使用容器化路径，可直接跳过本节，改用上一节的容器脚本。

只有在第 3 步的 compare env 脚本成功完成，或者你使用了官方容器 / 已验证专用环境时，才应该继续这一步。

```bash
cd /home/user8/sagellm
./scripts/sagellm_with_ascend_env.sh \
  python \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  > /tmp/vllm_ascend_8000.log 2>&1
```

说明：

- 必须通过 `sagellm_with_ascend_env.sh` 注入环境
- 首次启动可能需要等待模型下载与 ACL graph capture

判活要求：

```bash
ss -ltnp | grep ':8000'
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

## 8. 启动 `sagellm`

用于 benchmark 时，必须显式开启 `--benchmark-mode`，避免 canary 影响启动与测量：

```bash
cd /home/user8/sagellm
HF_ENDPOINT=https://hf-mirror.com \
SAGELLM_ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/8.3.RC2 \
./scripts/sagellm_with_ascend_env.sh \
  sagellm serve \
  --backend ascend \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8901 \
  --benchmark-mode \
  > /tmp/sagellm_8901.log 2>&1
```

判活要求：

```bash
ss -ltnp | grep ':8901'
curl http://127.0.0.1:8901/health
curl http://127.0.0.1:8901/v1/models
```

## 9. 运行对比 benchmark

```bash
BATCH_SIZES=1,2,4 MAX_OUTPUT_TOKENS=64 REQUEST_TIMEOUT=180 \
sagellm-benchmark compare \
  --target vllm=http://127.0.0.1:8000/v1 \
  --target sagellm=http://127.0.0.1:8901/v1 \
  --hardware-family ascend \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

不再推荐也不再保留旧的 shell compare wrapper；统一使用上面的 canonical CLI。

输出目录默认在：

```text
benchmark_results/compare_<timestamp>/
```

包含：

- `sagellm.json`
- `sagellm.md`
- `vllm.json`
- `vllm.md`
- `comparison.md`
- `comparison.json`

## 10. 当前状态

当前仓库已经完成的工作是：

- `sagellm` 主环境的 Ascend 启动链路已修通
- `sagellm-benchmark` 已提供 compare env 预检脚本
- `sagellm-benchmark` 已提供 `vllm-ascend` 官方容器启动脚本，适合作为旧 CANN 主机上的优先对比路径
- compare env 脚本现在会先验证完整 `vllm + vllm-ascend` 栈是否可解
- 对于当前这台机器上的旧 `0.11.0 + torch 2.7.1` 本地栈，脚本会明确报告 resolver 冲突，而不是继续误装

尚未被当前仓库宣称为默认已验证的内容：

- “在本机当前主环境中，直接一键安装完整 `vllm` endpoint compare 栈”
- “当前文档中的某组性能数字可作为新的默认基线”

## 11. 注意事项

- 不要把华为 Ascend 安装包（`.run` / `.deb`）提交到仓库；它们体积大，且通常应按上游许可与分发方式获取
- 推荐把“版本矩阵 + 下载地址 + 脚本”固化到文档和脚本，而不是提交二进制包
- 若 `vllm-ascend` 首次启动慢，优先等待模型下载和 ACL graph capture 完成，不要过早判定失败
- 若 `sagellm` benchmark 启动失败，优先确认是否遗漏 `--benchmark-mode`
- 若 compare env 脚本在 dry-run 阶段失败，优先调整到官方同版本矩阵，或改用官方 `vllm-ascend` 容器，而不是强行污染主 `sagellm` 环境
- 若脚本在 profile 选择阶段因主机 CANN 版本不匹配而失败，这是预期保护；当前脚本不会把 `8.1.x` 这类旧环境伪装成“官方 endpoint benchmark 可支持环境”。
