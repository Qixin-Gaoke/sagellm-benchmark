# 改进的 Benchmark 数据同步方案

## 🎯 设计目标

1. **用户无感知**：运行 benchmark 后自动上传，无需手动操作
2. **不污染 git**：原始数据不提交到 git，保持仓库轻量
3. **数据持久化**：所有数据安全存储在 HF，随时可查询
4. **可选本地保留**：用户可选择是否保留本地副本

---

## 🚀 方案 1：本地自动上传（推荐）

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户本地机器                              │
│                                                              │
│  $ sagellm-benchmark run --model gpt2 --backend cpu         │
│                                                              │
│  ↓ (benchmark 运行中...)                                     │
│                                                              │
│  outputs/cpu/gpt2/xxx/*_leaderboard.json  ← 临时生成         │
│                                                              │
│  ↓ (运行完成后自动触发)                                       │
│                                                              │
│  1. 读取 ~/.sagellm/config.yaml 获取 HF_TOKEN               │
│  2. 聚合本地结果 (aggregate_for_hf.py)                       │
│  3. 从 HF 下载现有数据并合并                                  │
│  4. 上传到 HF (upload_to_hf.py)                              │
│  5. (可选) 删除本地临时文件                                   │
│                                                              │
│  ✅ 完成！无需 git 操作                                       │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ HF API (自动)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              🤗 Hugging Face Datasets Hub                    │
│                                                              │
│  wangyao36/sagellm-benchmark-results                        │
│  ├── leaderboard_single.json  ← 自动更新                     │
│  └── leaderboard_multi.json                                 │
└─────────────────────────────────────────────────────────────┘
```

### 实现步骤

#### Step 1: 配置文件（首次设置）

```bash
# 用户首次使用时配置（一次性）
$ sagellm-benchmark config --hf-token hf_xxxxx

# 自动创建 ~/.sagellm/config.yaml
```

**配置文件内容**：
```yaml
# ~/.sagellm/config.yaml
huggingface:
  token: hf_xxxxxxxxxxxxxxxxxxxxx
  repo: wangyao36/sagellm-benchmark-results
  auto_upload: true  # 默认自动上传

local:
  keep_outputs: false  # 上传后删除本地文件（可选）
  outputs_dir: ~/sagellm-benchmark-outputs
```

#### Step 2: 修改 benchmark 运行脚本

```python
# src/sagellm_benchmark/cli.py

@click.command()
@click.option("--model", required=True)
@click.option("--backend", default="cpu")
@click.option("--auto-upload/--no-auto-upload", default=None)  # 可覆盖配置
def run(model: str, backend: str, auto_upload: bool | None):
    """运行 benchmark"""

    # 1. 运行 benchmark
    results = run_benchmark(model, backend)

    # 2. 保存本地结果
    save_results(results)

    # 3. 读取配置
    config = load_config()
    should_upload = auto_upload if auto_upload is not None else config.get("huggingface.auto_upload", True)

    # 4. 自动上传到 HF
    if should_upload:
        try:
            print("\n📤 自动上传到 Hugging Face...")
            upload_to_huggingface(results, config)
            print("✅ 上传成功！")

            # 5. 可选：清理本地文件
            if not config.get("local.keep_outputs", False):
                cleanup_local_outputs()
                print("🗑️  已清理本地临时文件")
        except Exception as e:
            print(f"⚠️  上传失败: {e}")
            print("💡 结果已保存到本地，可稍后手动上传")
```

#### Step 3: 上传函数实现

```python
# src/sagellm_benchmark/upload.py

def upload_to_huggingface(results: dict, config: dict) -> None:
    """自动聚合并上传到 HF"""

    # 1. 登录 HF
    token = config["huggingface"]["token"]
    repo = config["huggingface"]["repo"]
    login(token=token)

    api = HfApi()

    # 2. 下载现有数据
    existing_single = download_from_hf(repo, "leaderboard_single.json")
    existing_multi = download_from_hf(repo, "leaderboard_multi.json")

    # 3. 合并数据（智能去重）
    new_results = [results]  # 当前运行的结果
    merged_single, merged_multi = merge_with_existing(
        existing_single, existing_multi, new_results
    )

    # 4. 上传
    upload_leaderboard(api, repo, "leaderboard_single.json", merged_single)
    upload_leaderboard(api, repo, "leaderboard_multi.json", merged_multi)
```

### 用户体验

```bash
# 首次使用（配置 token）
$ sagellm-benchmark config --hf-token hf_xxxxx
✅ 配置已保存到 ~/.sagellm/config.yaml

# 运行 benchmark（自动上传）
$ sagellm-benchmark run --model gpt2 --backend cpu

Running benchmark...
✅ Benchmark completed!
  - TTFT: 45.2ms
  - Throughput: 80.0 tps

📤 自动上传到 Hugging Face...
  ✓ 下载现有数据 (12 条记录)
  ✓ 合并新结果 (新增 1 条)
  ✓ 上传到 wangyao36/sagellm-benchmark-results
✅ 上传成功！

🗑️  已清理本地临时文件

🔗 查看结果：
  https://huggingface.co/datasets/wangyao36/sagellm-benchmark-results
```

### 优点

✅ **用户无感知**：运行 benchmark 后自动上传，一气呵成  
✅ **不污染 git**：outputs/ 默认在 .gitignore，不提交  
✅ **数据安全**：所有数据存储在 HF，永久保存  
✅ **可选配置**：可关闭自动上传，或保留本地副本  
✅ **失败容错**：上传失败时本地数据仍保留

### 缺点

⚠️ 需要用户配置 HF_TOKEN（首次一次性）  
⚠️ 依赖网络连接

---

## 🚀 方案 2：后台服务收集（最无感知）

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户本地机器                              │
│                                                              │
│  $ sagellm-benchmark run --model gpt2 --backend cpu         │
│                                                              │
│  ↓ (运行完成后)                                              │
│                                                              │
│  POST https://sagellm-api.sage.org.ai/benchmark/submit      │
│  {                                                           │
│    "hardware": {...},                                        │
│    "model": {...},                                           │
│    "metrics": {...}                                          │
│  }                                                           │
│                                                              │
│  ✅ 完成！无需配置                                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ HTTPS POST (自动)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Backend Service (sagellm-api)                   │
│                                                              │
│  1. 接收 benchmark 结果                                      │
│  2. 验证数据格式                                              │
│  3. 与 HF 现有数据合并                                        │
│  4. 上传到 HF                                                │
│  5. 返回确认                                                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              🤗 Hugging Face Datasets Hub                    │
│                                                              │
│  wangyao36/sagellm-benchmark-results                        │
└─────────────────────────────────────────────────────────────┘
```

### 实现

```python
# src/sagellm_benchmark/cli.py

@click.command()
def run(model: str, backend: str):
    # 1. 运行 benchmark
    results = run_benchmark(model, backend)

    # 2. 自动提交到后台服务
    try:
        submit_to_backend(results)
        print("✅ 结果已自动提交！")
    except Exception as e:
        print(f"⚠️  提交失败（离线模式）: {e}")

def submit_to_backend(results: dict) -> None:
    """提交到后台服务"""
    url = "https://sagellm-api.sage.org.ai/benchmark/submit"

    # 匿名提交（或可选：附带用户 ID）
    response = requests.post(url, json=results, timeout=10)
    response.raise_for_status()
```

### 优点

✅ **完全无感知**：无需任何配置，运行即提交  
✅ **无需 token**：后台服务统一管理 HF_TOKEN  
✅ **数据验证**：服务端可验证数据完整性  
✅ **可扩展**：可添加分析、统计、排行榜等功能

### 缺点

⚠️ 需要部署和维护后台服务  
⚠️ 依赖网络连接  
⚠️ 隐私考虑（硬件信息上传）

---

## 🚀 方案 3：混合方案（灵活可控）

### 架构

```python
# 支持 3 种模式

# 模式 1：自动上传（默认）
$ sagellm-benchmark run --model gpt2
→ 自动上传到 HF（需配置 token）

# 模式 2：离线模式
$ sagellm-benchmark run --model gpt2 --offline
→ 仅保存本地，不上传

# 模式 3：稍后上传
$ sagellm-benchmark run --model gpt2 --offline
$ sagellm-benchmark upload outputs/cpu/gpt2/xxx/
→ 手动触发上传
```

---

## 📊 方案对比

| 特性           | 方案1：本地上传 | 方案2：后台服务 | 当前方案：GitHub Actions |
|----------------|----------------|----------------|-------------------------|
| 用户操作       | 首次配置 token  | 完全无需配置    | git add/commit/push      |
| 网络依赖       | ✅ 需要         | ✅ 需要         | ✅ 需要                  |
| git 仓库大小   | ✅ 轻量         | ✅ 轻量         | ❌ 臃肿                  |
| 部署复杂度     | ✅ 简单         | ❌ 需要服务器   | ✅ 简单                  |
| 数据隐私       | ✅ 用户控制     | ⚠️ 服务器处理   | ✅ 用户控制              |
| 失败容错       | ✅ 本地保留     | ⚠️ 需重试机制   | ✅ git 中保留            |
| 推荐指数       | ⭐⭐⭐⭐⭐       | ⭐⭐⭐⭐        | ⭐⭐⭐                   |

---

## 🚨 关键问题：Token 安全性

### 问题

**不能将 HF_TOKEN 分发给每个测评用户！**

- HF_TOKEN 是私有密钥，具有写入权限
- 泄露后任何人都可以篡改数据
- 无法撤销单个用户的访问权限

### 解决方案对比

| 方案 | Token 管理 | 适用场景 | 安全性 |
|-----|-----------|---------|--------|
| 方案1（本地上传） | ❌ 每个用户需要 token | 仅核心团队 | ⚠️ 风险高 |
| 方案2（后台服务） | ✅ 服务端统一管理 | 所有用户 | ✅ 安全 |
| 方案3（GitHub Actions） | ✅ GitHub Secrets | 有 git 权限的用户 | ✅ 安全 |
| 方案4（混合） | ✅ 分层管理 | 灵活 | ✅ 安全 |

---

## 🎯 推荐实施方案（修订版）

### **方案 4：混合方案**（最佳实践）⭐⭐⭐⭐⭐

**核心思路**：根据用户角色采用不同策略

#### 角色 1：核心团队成员（有 GitHub 写权限）

```bash
# 工作流程
$ sagellm-benchmark run --model gpt2
$ git add outputs/
$ git commit -m "feat: add benchmark results"
$ git push

# GitHub Actions 自动上传到 HF
# Token 安全地存储在 GitHub Secrets
```

✅ 无需分发 HF_TOKEN  
✅ git 权限即可  
✅ 审计追踪（git log）

#### 角色 2：外部贡献者（无写权限）

```bash
# 工作流程
$ sagellm-benchmark run --model gpt2 --backend ascend
$ sagellm-benchmark export --format json > my_results.json

# 通过以下方式之一提交：
# 方式A: 提交 PR（包含 outputs/）
# 方式B: 提交 Issue（附带 my_results.json）
# 方式C: 提交到公共表单/API（如果有后台服务）
```

✅ 不需要任何 token  
✅ 团队审核后再上传  
✅ 防止恶意数据

#### 角色 3：研究人员（仅查看）

```bash
# 直接访问 Hugging Face 公开数据
$ pip install datasets
$ from datasets import load_dataset
$ data = load_dataset("wangyao36/sagellm-benchmark-results")
```

✅ 公开可访问  
✅ 无需任何权限

---

## 🚀 方案 4 详细设计

### 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                     核心团队成员                              │
│  (有 sagellm-benchmark GitHub 仓库写权限)                     │
│                                                              │
│  $ sagellm-benchmark run --model gpt2                       │
│  $ git add outputs/ && git commit && git push               │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   │ git push (触发 Actions)
                   ↓
┌──────────────────────────────────────────────────────────────┐
│              GitHub Actions (自动触发)                        │
│                                                              │
│  HF_TOKEN 存储在 GitHub Secrets ✅                            │
│                                                              │
│  1. 聚合 outputs/ 数据                                       │
│  2. 从 HF 下载现有数据                                        │
│  3. 智能合并                                                 │
│  4. 上传到 HF                                                │
│  5. (可选) 提交后删除 outputs/，保持仓库轻量                  │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────────────────────┐
│              🤗 Hugging Face (公开数据集)                     │
│                                                              │
│  wangyao36/sagellm-benchmark-results                        │
│  - 任何人可查看 ✅                                            │
│  - 只有 Actions 可写 🔒                                      │
└──────────────────────────────────────────────────────────────┘
                   │
                   │ 公开访问
                   ↓
┌──────────────────────────────────────────────────────────────┐
│                   外部贡献者 / 研究人员                        │
│                                                              │
│  Option A: Fork + PR (需审核)                                │
│  Option B: 下载数据用于研究                                   │
└──────────────────────────────────────────────────────────────┘
```

### 关键改进：自动清理 git 历史

**问题**：outputs/ 提交后会让 git 仓库膨胀

**解决**：GitHub Actions 上传后自动删除 outputs/

```yaml
# .github/workflows/upload-to-hf.yml

jobs:
  upload-to-hf:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      # ... (聚合和上传步骤)

      - name: Clean up outputs directory
        run: |
          # 上传成功后，删除 outputs/ 保持仓库轻量
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

          # 删除已上传的文件
          git rm -rf outputs/
          git commit -m "chore: cleanup outputs after HF upload [skip ci]" || true
          git push
```

**效果**：
- ✅ outputs/ 只在本地和 Actions 运行时存在
- ✅ git 历史中不保留大文件
- ✅ 仓库始终保持轻量

---

## 🔐 Token 安全最佳实践

### GitHub Secrets 配置

```bash
# 仓库管理员配置（一次性）
# Settings → Secrets and variables → Actions → New repository secret

Name: HF_TOKEN
Value: hf_xxxxxxxxxxxxxxxxxxxxxx
```

### Workflow 中使用

```yaml
- name: Upload to Hugging Face
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}  # 安全注入
  run: |
    python scripts/upload_to_hf.py
```

**安全保障**：
- ✅ Token 不出现在代码中
- ✅ Token 不出现在日志中（自动脱敏）
- ✅ 只有仓库管理员可见
- ✅ 支持轮换更新

---

## 🎯 最终推荐方案

### 当前阶段（核心团队测试）

**使用 GitHub Actions（当前方案）+ 自动清理**

```bash
# 用户操作
$ sagellm-benchmark run --model gpt2
$ git add outputs/
$ git commit -m "feat: add benchmark"
$ git push

# 自动完成：
# 1. Actions 上传到 HF
# 2. Actions 删除 outputs/
# 3. git 仓库保持轻量
```

### 未来扩展（外部贡献）

**选项 A：PR 审核模式**
- 外部用户 fork 仓库
- 提交 PR（包含 outputs/）
- 团队审核后 merge
- Actions 自动上传

**选项 B：后台服务（长期）**
- 部署公共 API
- 用户提交结果到 API
- 服务端验证 + 上传
- 无需分发任何 token

---

## � 方案 5：本地聚合 + Actions 上传（最优方案）⭐⭐⭐⭐⭐

### 核心思路

**用户侧**：
1. 从 HF 拉取公开数据（无需 token）
2. 与本地 outputs/ 合并
3. 生成 hf_data/（标准格式）
4. 只提交 hf_data/，不提交 outputs/

**GitHub Actions 侧**：
1. 再次从 HF 拉取最新数据
2. 与用户提交的 hf_data/ 合并（解决并发）
3. 上传到 HF

### 架构图

```
用户A本地                           用户B本地
    │                                  │
    ├─ 运行 benchmark                  ├─ 运行 benchmark
    │  → outputs/a.json                │  → outputs/b.json
    │                                  │
    ├─ 拉取 HF 数据 (v1)               ├─ 拉取 HF 数据 (v1)
    │  → existing: [E1, E2]            │  → existing: [E1, E2]
    │                                  │
    ├─ 本地合并                        ├─ 本地合并
    │  → hf_data/: [E1, E2, A]         │  → hf_data/: [E1, E2, B]
    │                                  │
    └─ git push (只含 hf_data/)        └─ git push (只含 hf_data/)
         │                                  │
         │                                  │
         ↓                                  ↓
┌────────────────────────────────────────────────────────────┐
│              GitHub Actions (并发安全合并)                  │
│                                                             │
│  用户A的 push:                                              │
│  1. 拉取 HF 最新数据 (v1): [E1, E2]                         │
│  2. 合并 hf_data/: [E1, E2, A]                              │
│  3. 上传到 HF (v2): [E1, E2, A]                             │
│                                                             │
│  用户B的 push (稍后):                                       │
│  1. 拉取 HF 最新数据 (v2): [E1, E2, A]  ← 包含A的数据！     │
│  2. 合并 hf_data/: [E1, E2, A, B]      ← B的数据追加       │
│  3. 上传到 HF (v3): [E1, E2, A, B]     ← 两者都保留！       │
└────────────────────────────────────────────────────────────┘
                          │
                          ↓
                🤗 Hugging Face (最终结果)
                [E1, E2, A, B] ← 所有数据都保留
```

### 解决并发冲突的关键

**问题**：两个用户基于同一版本（v1）提交数据

**解决**：GitHub Actions 作为"最终裁决者"，再次合并

```python
# .github/workflows/upload-to-hf.yml 的逻辑

def upload_with_conflict_resolution():
    # 1. 读取用户提交的数据
    user_data = load_json("hf_data/leaderboard_single.json")

    # 2. 从 HF 拉取最新数据（可能已被其他用户更新）
    latest_hf_data = download_from_hf("leaderboard_single.json")

    # 3. 三方合并（智能去重）
    merged_data = smart_merge(
        base=latest_hf_data,      # HF 最新版本（权威）
        incoming=user_data         # 用户提交的数据
    )

    # 4. 上传合并后的结果
    upload_to_hf(merged_data)
```

### 用户工作流

```bash
# 1. 运行 benchmark
$ sagellm-benchmark run --model gpt2 --backend cpu

# 2. 本地聚合（自动或手动）
$ sagellm-benchmark aggregate
📥 从 HF 下载最新数据...
  ✓ leaderboard_single.json (123 条)
  ✓ leaderboard_multi.json (45 条)

🔀 合并本地结果...
  ✓ 扫描 outputs/ (找到 3 个新结果)
  ✓ 智能去重
  ↑ 新增 2 条，更新 1 条

💾 保存到 hf_data/
  ✓ hf_data/leaderboard_single.json
  ✓ hf_data/leaderboard_multi.json

# 3. 提交（只提交 hf_data/，不提交 outputs/）
$ git add hf_data/
$ git commit -m "feat: add gpt2 benchmark results"
$ git push

# 4. GitHub Actions 自动处理并发，上传到 HF
✅ 完成！
```

### .gitignore 配置

```gitignore
# 忽略原始实验数据（不提交）
outputs/

# 提交聚合后的数据（标准格式）
!hf_data/
```

### GitHub Actions Workflow（改进版）

```yaml
# .github/workflows/upload-to-hf.yml

name: Upload to Hugging Face

on:
  push:
    branches:
      - main
      - main
    paths:
      - 'hf_data/**/*.json'  # 只监听 hf_data/ 变化

jobs:
  upload-to-hf:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install huggingface_hub

      # 关键步骤：并发安全合并
      - name: Merge with latest HF data (conflict resolution)
        env:
          HF_REPO: wangyao36/sagellm-benchmark-results
        run: |
          python scripts/merge_and_upload.py

      - name: Upload to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_REPO: wangyao36/sagellm-benchmark-results
        run: |
          python scripts/upload_to_hf.py

      # 可选：上传成功后清理
      - name: Cleanup hf_data (keep repo clean)
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git rm -rf hf_data/
          git commit -m "chore: cleanup hf_data after upload [skip ci]" || true
          git push
```

### 关键脚本：merge_and_upload.py

```python
#!/usr/bin/env python3
"""
并发安全的合并和上传脚本

关键逻辑：
1. 读取用户提交的 hf_data/
2. 从 HF 下载最新数据（可能已被其他用户更新）
3. 三方智能合并
4. 上传到 HF
"""

from __future__ import annotations
import json
from pathlib import Path
from huggingface_hub import HfApi
import urllib.request

HF_REPO = "wangyao36/sagellm-benchmark-results"

def download_from_hf(filename: str) -> list[dict]:
    """从 HF 下载最新数据"""
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except:
        return []

def get_config_key(entry: dict) -> str:
    """生成唯一标识（用于去重）"""
    hw = entry.get("hardware", {})
    model = entry.get("model", {})
    workload = entry.get("workload", {})

    return "|".join([
        hw.get("chip_model", "unknown"),
        str(hw.get("chip_count", 1)),
        model.get("name", "unknown"),
        model.get("precision", "FP16"),
        str(workload.get("input_length", 0)),
        str(workload.get("output_length", 0)),
    ])

def is_better_result(new: dict, old: dict) -> bool:
    """判断新结果是否更优"""
    new_tps = new.get("metrics", {}).get("throughput_tps", 0)
    old_tps = old.get("metrics", {}).get("throughput_tps", 0)
    return new_tps > old_tps

def smart_merge(hf_latest: list[dict], user_data: list[dict]) -> list[dict]:
    """
    三方智能合并

    规则：
    1. HF 最新数据为基准（权威）
    2. 用户数据追加或更新
    3. 相同配置时，选择性能更好的
    """
    merged = {}

    # 先加入 HF 最新数据（权威）
    for entry in hf_latest:
        key = get_config_key(entry)
        merged[key] = entry

    # 合并用户数据
    added = 0
    updated = 0

    for entry in user_data:
        key = get_config_key(entry)

        if key not in merged:
            merged[key] = entry
            added += 1
            print(f"  ✓ 新增: {key[:60]}")
        else:
            if is_better_result(entry, merged[key]):
                merged[key] = entry
                updated += 1
                print(f"  ↑ 更新: {key[:60]}")

    print(f"\n📊 合并结果: 新增 {added}, 更新 {updated}, 总计 {len(merged)}")
    return list(merged.values())

def main():
    print("🔀 并发安全合并...")

    # 1. 读取用户提交的数据
    hf_data_dir = Path("hf_data")
    user_single = json.loads((hf_data_dir / "leaderboard_single.json").read_text())
    user_multi = json.loads((hf_data_dir / "leaderboard_multi.json").read_text())

    # 2. 从 HF 下载最新数据（可能已被其他用户更新）
    print("\n📥 从 HF 下载最新数据...")
    hf_single = download_from_hf("leaderboard_single.json")
    hf_multi = download_from_hf("leaderboard_multi.json")
    print(f"  ✓ Single: {len(hf_single)} 条")
    print(f"  ✓ Multi: {len(hf_multi)} 条")

    # 3. 智能合并
    print("\n🔀 合并数据...")
    merged_single = smart_merge(hf_single, user_single)
    merged_multi = smart_merge(hf_multi, user_multi)

    # 4. 保存合并结果（覆盖 hf_data/）
    (hf_data_dir / "leaderboard_single.json").write_text(
        json.dumps(merged_single, indent=2, ensure_ascii=False)
    )
    (hf_data_dir / "leaderboard_multi.json").write_text(
        json.dumps(merged_multi, indent=2, ensure_ascii=False)
    )

    print("\n✅ 合并完成！准备上传...")

if __name__ == "__main__":
    main()
```

### 本地聚合脚本（用户使用）

```python
# src/sagellm_benchmark/cli.py

@click.command()
def aggregate():
    """聚合本地结果并准备上传"""

    # 1. 从 HF 下载最新数据（公开，无需 token）
    print("📥 从 Hugging Face 下载最新数据...")
    hf_single = download_from_hf("leaderboard_single.json")
    hf_multi = download_from_hf("leaderboard_multi.json")

    # 2. 扫描本地 outputs/
    print("\n📂 扫描本地结果...")
    local_results = scan_outputs_dir()

    # 3. 合并
    print("\n🔀 合并数据...")
    merged_single, merged_multi = merge_results(
        hf_single, hf_multi, local_results
    )

    # 4. 保存到 hf_data/
    hf_data_dir = Path("hf_data")
    hf_data_dir.mkdir(exist_ok=True)

    save_json(hf_data_dir / "leaderboard_single.json", merged_single)
    save_json(hf_data_dir / "leaderboard_multi.json", merged_multi)

    print("\n✅ 聚合完成！")
    print(f"  📄 hf_data/leaderboard_single.json ({len(merged_single)} 条)")
    print(f"  📄 hf_data/leaderboard_multi.json ({len(merged_multi)} 条)")
    print("\n💡 下一步:")
    print("  git add hf_data/")
    print("  git commit -m 'feat: add benchmark results'")
    print("  git push")
```

---

## 📊 方案对比（最终版）

| 特性 | 原方案 | 方案5（推荐） |
|-----|-------|-------------|
| 提交内容 | outputs/ (大量文件) | hf_data/ (2个文件) |
| git 仓库大小 | ❌ 膨胀 | ✅ 轻量 |
| 并发冲突 | ⚠️ 可能冲突 | ✅ Actions 自动解决 |
| Token 安全 | ✅ 集中管理 | ✅ 集中管理 |
| 用户操作 | git add/commit/push | aggregate + git push |
| 数据丢失风险 | ❌ 可能（并发） | ✅ 无（智能合并） |
| **推荐指数** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 核心优势

### 1. 解决并发冲突 ✅

```
用户A push → Actions 基于 HF v1 合并 → 上传 v2
用户B push → Actions 基于 HF v2 合并 → 上传 v3 (包含A+B)
```

### 2. git 仓库轻量 ✅

```
outputs/     → 不提交 (在 .gitignore)
hf_data/     → 提交 (仅2个JSON，几KB)
             → Actions 上传后自动删除 (可选)
```

### 3. 无需分发 Token ✅

```
用户：拉取 HF 数据（公开，无需 token）
Actions：上传 HF 数据（使用 GitHub Secrets）
```

### 4. 数据永不丢失 ✅

```
Actions 总是基于 HF 最新版本合并
→ 智能去重，性能更优者胜出
→ 不同配置追加
```

---

## �🛠️ 实施步骤

### Phase 1: 核心功能（1-2 天）

1. ✅ 创建配置管理模块（`config.py`）
2. ✅ 实现自动上传逻辑（`upload.py`）
3. ✅ 集成到 CLI（`cli.py`）
4. ✅ 添加 `--offline` 选项

### Phase 2: 用户体验优化（1 天）

1. ✅ 友好的首次配置流程
2. ✅ 上传进度提示
3. ✅ 失败时的降级处理
4. ✅ 文档和示例

### Phase 3: 文档和测试（1 天）

1. ✅ 更新 README
2. ✅ 编写测试用例
3. ✅ 用户指南

---

## 📝 Git 仓库清理

实施新方案后，可以清理 git 历史：

```bash
# 1. 从 git 历史中移除 outputs/ 目录
git filter-branch --force --index-filter \
  "git rm -rf --cached --ignore-unmatch outputs/" \
  --prune-empty --tag-name-filter cat -- --all

# 2. 强制推送（慎重！）
git push origin --force --all

# 3. 清理本地引用
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

**注意**：这会重写 git 历史，谨慎操作！

---

## 🎉 总结

新方案的核心优势：

1. **用户友好**：配置一次，永久生效
2. **仓库轻量**：不再保存大量原始数据
3. **数据安全**：HF 作为永久存储
4. **灵活可控**：支持离线模式和手动上传

这样就实现了你想要的"丝滑"体验！🚀
