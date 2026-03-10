# Benchmark Clients Guide

## ⚠️ 重要说明：生产环境 vs 基准测试

### sageLLM 生产环境使用方式

**对外服务（生产环境）推荐架构**：

```
外部用户
    ↓ (HTTP API - OpenAI 协议)
┌─────────────────────────────────┐
│   sagellm-gateway (L5)          │  ← 必须通过 Gateway（HTTP API）
│   - OpenAI 兼容 API             │
│   - 会话管理                    │
│   - 认证授权                    │
└─────────────────────────────────┘
    ↓ (内部调用)
┌─────────────────────────────────┐
│   Control Plane (L4)            │  ← 调度、负载均衡
│   - 请求路由                    │
│   - 引擎调度                    │
└─────────────────────────────────┘
    ↓ (内部调用)
┌─────────────────────────────────┐
│   sagellm-core + backend (L1-L3)│  ← 推理引擎
│   - 实际推理执行                │
└─────────────────────────────────┘
```

**✅ 生产环境使用方式**：

1. **对外 HTTP API（推荐）**：
   ```bash
   # 一键启动完整服务
   pip install 'isagellm[server]'
   sage-llm serve --model Qwen2-7B --port 8000

   # 外部用户通过 OpenAI 协议访问
   curl http://your-server:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen2-7B", "messages": [...]}'
   ```

2. **Python API（内部集成）**：
   ```python
   # 仅用于 Python 应用内部集成，不对外暴露
   from sagellm import ControlPlaneManager

   manager = ControlPlaneManager(...)
   result = await manager.route_request(request)
   ```

**❌ 不推荐**：直接调用 sagellm-backend（跳过 Control Plane）

### 基准测试场景

sagellm-benchmark 的客户端是为了 **性能测试**，不是生产部署：

### 依赖约定

- `pyproject.toml` 里的 benchmark extras 是对比客户端依赖的唯一声明来源。
- 安装脚本只负责便捷地铺设或固定一套已验证环境，不替代 extras。
- 需要第三方对比引擎时，先安装对应 extras；脚本只在你需要复现特定环境矩阵时再额外执行。
- `./quickstart.sh` 会按当前机器类型自动补装匹配的 compare extra；在 Ascend 上还会额外叠加一套已验证版本矩阵，但该矩阵仍从属于 extras，而不是新的依赖入口。

```bash
# vLLM 对比（通用环境）
pip install -U 'isagellm-benchmark[vllm-client]'

# vLLM Ascend 对比（Ascend 机器）
pip install -U 'isagellm-benchmark[vllm-ascend-client]'

# LMDeploy 对比
pip install -U 'isagellm-benchmark[lmdeploy-client]'

# 仅在需要复现已验证的 Ascend 版本矩阵时，再运行便利脚本
bash scripts/setup_vllm_ascend_compare_env.sh
```

如果你直接使用 quickstart：

```bash
./quickstart.sh --dev
```

行为约定：

- 非 Ascend 机器：安装 `isagellm-benchmark[vllm-client]`
- Ascend 机器：安装 `isagellm-benchmark[vllm-ascend-client]`，再叠加已验证版本矩阵

## Overview

sagellm-benchmark 提供多种客户端，用于测试不同的 LLM 服务：

| 客户端 | 用途 | 连接方式 |
|--------|------|---------|
| **GatewayClient** | sagellm-gateway（OpenAI 协议） | HTTP API |
| **SageLLMClient** | sagellm-backend 原生引擎 | 直接调用（无 HTTP） |
| **VLLMClient** | vLLM | HTTP API |
| **LMDeployClient** | LMDeploy | HTTP API |

## 唯一对比入口

跨引擎对比现在统一收敛到 `sagellm-benchmark`：

1. 优先使用 `sagellm-benchmark compare` 对多个 OpenAI-compatible endpoint 做 live 对比。
2. 对于标准 `sageLLM vs vLLM` 流程，优先使用更薄的 `sagellm-benchmark vllm-compare run` 包装入口。
3. 若服务没有兼容 OpenAI 的 `/v1` 接口，再使用 benchmark Python client（如 `LMDeployClient`）补齐。
4. `sagellm-core` 只保留 SageLLM 自身引擎能力与通用插件抽象，不再作为 vLLM/LMDeploy 对比入口。

标准 `sageLLM vs vLLM` 推荐 CLI：

```bash
sagellm-benchmark vllm-compare run \
    --sagellm-url http://127.0.0.1:8901/v1 \
    --vllm-url http://127.0.0.1:8000/v1 \
    --model Qwen/Qwen2.5-0.5B-Instruct
```

在 A100/CUDA 机器上，推荐把 vLLM 固定为独立 Docker 服务，而不是每次在宿主机里重新安装 / 重配 `vllm` wheel：

```bash
cd sagellm-benchmark
VLLM_GPU_DEVICE=1 VLLM_PORT=9100 ./scripts/start_vllm_cuda_docker.sh

sagellm-benchmark vllm-compare run \
    --sagellm-url http://127.0.0.1:8901/v1 \
    --vllm-url http://127.0.0.1:9100/v1 \
    --model Qwen/Qwen2.5-1.5B-Instruct
```

如果希望 compare 在 endpoint 缺失时自动把容器拉起：

```bash
sagellm-benchmark vllm-compare run \
    --sagellm-url http://127.0.0.1:8901/v1 \
    --vllm-url http://127.0.0.1:9100/v1 \
    --start-vllm-cmd "./scripts/start_vllm_cuda_docker.sh" \
    --model Qwen/Qwen2.5-1.5B-Instruct
```

默认容器名为 `sagellm-benchmark-vllm`，停止时执行：

```bash
./scripts/stop_vllm_cuda_docker.sh
```

默认情况下该 helper 不会给容器加 `--rm`，这样遇到 OOM、模型加载失败或参数不兼容时，可以直接查看容器日志：

```bash
docker logs sagellm-benchmark-vllm | tail -n 200
```

该 helper 默认使用 `--network host`，优先规避受限服务器上 Docker bridge 无法访问外网仓库的问题；只有明确需要端口映射隔离时，再通过 `VLLM_DOCKER_NETWORK_MODE=bridge` 切回 bridge 模式。

推荐 CLI：

```bash
sagellm-benchmark compare \
    --target sagellm=http://127.0.0.1:8902/v1 \
    --target vllm=http://127.0.0.1:8901/v1 \
    --target lmdeploy=http://127.0.0.1:23333/v1 \
    --model Qwen/Qwen2.5-0.5B-Instruct
```

该命令会在 benchmark 侧统一完成：

- endpoint 健康检查与模型发现
- TTFT/TBT/TPS live 指标采集
- 每个 target 的 JSON/Markdown 产物输出
- 汇总 `comparison.json` 与 `comparison.md`

## GatewayClient（推荐用于完整系统测试）

### 什么是 GatewayClient？

`GatewayClient` 是用于连接 **OpenAI 协议** HTTP API 的客户端。它主要用于：

1. **测试 sagellm-gateway** - sageLLM 的 API 网关服务
2. **对比 OpenAI API** - 可连接 OpenAI 官方 API 进行性能对比
3. **测试其他兼容服务** - vLLM OpenAI server、LMDeploy OpenAI server 等

### 为什么不叫 OpenAIClient？

之前叫 `OpenAIClient`，但容易混淆：
- ❌ 用户误以为是 OpenAI 公司专用
- ❌ 不清楚可以连接 sagellm-gateway
- ✅ 改名为 `GatewayClient` 更清晰

### 使用示例

```python
from sagellm_benchmark.clients.openai_client import GatewayClient
from sagellm_benchmark.types import BenchmarkRequest

# 连接 sagellm-gateway（主要用途）
client = GatewayClient(
    base_url="http://localhost:8000/v1",
    api_key="benchmark",
)

# 创建请求
request = BenchmarkRequest(
    prompt="Hello!",
    max_tokens=100,
)

# 执行
result = await client.generate(request)
print(f"TTFT: {result.metrics.ttft_ms:.2f}ms")
```

### compare 命令默认就走 GatewayClient

对于 `sagellm`、`vllm`、`lmdeploy` 这类已经暴露 OpenAI-compatible endpoint 的服务，`compare` 命令默认使用 `GatewayClient` 统一测量，而不是在每个引擎 client 中复制一套 HTTP/streaming/metrics 逻辑。

因此：

- `VLLMClient(server)` 主要用于 benchmark 内部复用或特殊脚本场景
- 正式跨引擎 live 对比默认仍以 `compare` 为准

### 对比：OpenAI 官方 API

```python
# 也可以连接 OpenAI 官方 API（用于性能对比）
openai_client = GatewayClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",  # 真实 API key
)
```

## SageLLMClient（用于原生后端测试）

### 什么是 SageLLMClient？

`SageLLMClient` 直接使用 **sagellm-backend** 原生引擎，**不通过 HTTP**。

适用场景：
- 测试引擎核心性能（无网络开销）
- 本地开发和调试
- 单元测试

注意：`SageLLMClient` 用于本地原生引擎测试，不是跨引擎对比的首选入口。跨引擎对比优先走 `compare` 或 `GatewayClient`。

### 使用示例

```python
from sagellm_benchmark.clients.sagellm_client import SageLLMClient
from sagellm_backend import CPUEngine

# 创建原生引擎
engine = CPUEngine(
    model_path="gpt2",
    device="cpu",
)

# 包装为 benchmark 客户端
client = SageLLMClient(engine=engine)

# 使用方式相同
result = await client.generate(request)
```

## 客户端选择指南

| 测试场景 | 推荐客户端 | 原因 |
|---------|-----------|------|
| sagellm vs vllm/lmdeploy 标准对比 | **`sagellm-benchmark compare`** | benchmark 唯一正式入口，统一 endpoint 探活、live 指标和输出 |
| 完整系统性能测试（生产架构）| **GatewayClient** | 包含完整调用链路（HTTP + Control Plane），符合生产部署 |
| API 网关功能测试 | **GatewayClient** | 直接测试 sagellm-gateway |
| 引擎核心性能测试（无网络开销）| **SageLLMClient** | 绕过网络层，聚焦引擎性能 |
| 与 OpenAI 对比 | **GatewayClient** | 可连接 OpenAI API |
| vLLM 对比测试 | **GatewayClient / compare** | vLLM server mode 已复用通用 OpenAI-compatible client |
| LMDeploy 对比测试 | **GatewayClient / LMDeployClient** | 先用 OpenAI-compatible endpoint；仅在非兼容接口时再走专用 client |

### 关键原则

1. **生产环境必须走 HTTP**：外部用户只能通过 sagellm-gateway（OpenAI API）访问
2. **跨引擎对比统一走 benchmark**：使用 `compare` 或 benchmark client，不在 core 里扩第三方 adaptor
3. **基准测试模拟真实场景**：使用 GatewayClient 测试完整调用链路
4. **引擎性能测试可直连**：使用 SageLLMClient 排除网络干扰

## 架构说明

### 生产环境完整架构

```
外部用户（OpenAI SDK/curl）
    ↓ HTTP (OpenAI 协议)
┌───────────────────────────────────────┐
│  sagellm-gateway (API Gateway)        │
│  • OpenAI 兼容 endpoints              │
│  • 认证授权                            │
│  • 会话管理                            │
└───────────────────────────────────────┘
    ↓ 内部调用
┌───────────────────────────────────────┐
│  Control Plane (调度层)                │
│  • 请求路由                            │
│  • 负载均衡                            │
│  • 引擎生命周期管理                     │
└───────────────────────────────────────┘
    ↓ 内部调用
┌───────────────────────────────────────┐
│  sagellm-core + backend (引擎层)      │
│  • 实际推理执行                        │
│  • KV 缓存管理                         │
│  • 硬件加速（CPU/CUDA/Ascend）         │
└───────────────────────────────────────┘
```

### 基准测试客户端架构

```
┌─────────────────────────────────────────────────────────────┐
│                  sagellm-benchmark                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ GatewayClient│  │ SageLLMClient│  │  VLLMClient  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │ HTTP            │ Direct          │ HTTP          │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│ sagellm-gateway │  │  sagellm-    │  │     vLLM     │
│  (生产架构)     │  │   backend    │  │    Server    │
│                 │  │ （引擎直连）  │  │              │
└─────────────────┘  └──────────────┘  └──────────────┘

说明：
• GatewayClient：测试生产环境完整链路（推荐）
• SageLLMClient：测试引擎性能（排除网络开销）
```

## 常见问题

### Q: sageLLM 对外使用时，是直连还是走 HTTP？

**A:** **必须走 HTTP**（sagellm-gateway）。

生产环境架构：
```
外部用户 → HTTP API (sagellm-gateway) → Control Plane → 引擎
```

原因：
1. ✅ 标准化接口：OpenAI 兼容协议，易于集成
2. ✅ 认证授权：Gateway 处理 API key、限流
3. ✅ 会话管理：多轮对话状态维护
4. ✅ 负载均衡：Control Plane 自动调度请求

**直连引擎仅用于**：
- 内部开发调试
- 性能基准测试（排除网络开销）
- Python 应用内嵌集成（非对外服务）

### Q: 为什么基准测试有两个客户端（GatewayClient 和 SageLLMClient）？

**A:**
- **GatewayClient**：测试完整系统（含 HTTP + Control Plane），符合真实部署
- **SageLLMClient**：测试引擎核心，排除网络干扰

两者测试不同层次的性能。

### Q: GatewayClient 能连接 OpenAI 官方 API 吗？

**A:** 可以！只需改变 `base_url` 和 `api_key`：

```python
GatewayClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-YOUR_KEY",
)
```

### Q: 我应该用哪个客户端测试 sageLLM？

**A:** 优先使用 **GatewayClient + sagellm-gateway**，因为：
- 完整的调用链路（包括 Control Plane）
- 符合生产部署架构
- 可测试 API 网关功能

## 迁移指南（旧代码）

如果你的代码使用了旧的 `OpenAIClient`，请更新：

```python
# ❌ 旧代码
from sagellm_benchmark.clients.openai_client import OpenAIClient
client = OpenAIClient(...)

# ✅ 新代码
from sagellm_benchmark.clients.openai_client import GatewayClient
client = GatewayClient(...)
```

功能完全相同，只是名字更清晰。
