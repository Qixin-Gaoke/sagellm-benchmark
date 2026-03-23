# sageLLM Benchmark 工作流程

## 📝 简介

本文档说明如何提交 benchmark 结果到 HuggingFace，以及自动化流程如何工作。

---

## 👤 用户需要做的事情（3 步）

### 1️⃣ 运行 Benchmark
```bash
# 在本地运行性能测试
sagellm-benchmark run --model Qwen2-7B

# 结果保存在 outputs/ 目录（不提交到 Git）
```

### 2️⃣ 聚合数据
```bash
# 将 outputs/ 的结果聚合到 hf_data/
python scripts/aggregate_for_hf.py

# 或使用命令
sagellm-benchmark aggregate
```

### 3️⃣ 提交到 GitHub
```bash
# 只提交 hf_data/ 目录
git add hf_data/
git commit -m "feat: add Qwen2-7B benchmark results"
git push origin main
```

**✅ 完成！后续都是自动的。**

---

## 🤖 自动化流程（无需手动操作）

推送 `hf_data/*.json` 到 GitHub 后，GitHub Actions 会自动执行：

### Step 1: 从 HuggingFace 拉取最新数据
```
从 https://huggingface.co/datasets/intellistream/sagellm-benchmark-results
下载 leaderboard_single.json 和 leaderboard_multi.json
```

### Step 2: 智能合并
```
合并规则：
- 相同配置的测试结果只保留最优的（throughput 最高）
- 不同配置的结果都保留
- 避免数据丢失
```

**并发安全**：即使多个用户同时提交，数据也不会丢失。

### Step 3: 上传到 HuggingFace
```
上传合并后的数据到 HuggingFace
仓库：intellistream/sagellm-benchmark-results
```

### Step 4: 清理 Git 仓库
```
删除 hf_data/ 目录（已上传到 HF，无需保留在 Git）
保持仓库轻量
```

### ⚠️ 当 GitHub Actions 因账单/配额被阻塞时

如果出现类似 “job was not started because recent account payments have failed or spending limit needs to be increased” 的提示，
说明是平台账单限制，不是仓库代码错误。此时可先执行本地兜底流程，避免开发停滞：

```bash
# 在仓库根目录执行
bash scripts/local_ci_fallback.sh
```

该脚本会按 `ci.yml` 的核心顺序运行：
- pre-commit 全量检查
- version source guard
- pytest + coverage（`--cov-fail-under=45`）
- build + twine check

---

## 🔄 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│ 本地操作（用户手动）                                          │
└─────────────────────────────────────────────────────────────┘

  运行 Benchmark
       ↓
  outputs/result.json ─────────────────┐
                                       │
  聚合数据                              │ (不提交到 Git)
       ↓                               │
  hf_data/leaderboard_*.json ←────────┘
       ↓
  git add hf_data/
  git commit
  git push
       ↓

┌─────────────────────────────────────────────────────────────┐
│ GitHub Actions (自动)                                        │
└─────────────────────────────────────────────────────────────┘

  触发条件: hf_data/*.json 被推送
       ↓
  1. 从 HF 下载最新数据
       ↓
  2. 智能合并（你的数据 + HF 最新数据）
       ↓
  3. 上传到 HuggingFace
       ↓
  4. 删除 Git 中的 hf_data/
       ↓
  ✅ 完成

┌─────────────────────────────────────────────────────────────┐
│ 查看结果                                                     │
└─────────────────────────────────────────────────────────────┘

访问 HuggingFace 数据集：
https://huggingface.co/datasets/intellistream/sagellm-benchmark-results
```

---

## ❓ 常见问题

### Q1: 为什么要先聚合再提交？
**A**: 聚合脚本会从 HuggingFace 下载最新数据并与你的结果合并，避免覆盖别人的数据。

### Q2: 多人同时提交会冲突吗？
**A**: 不会。GitHub Actions 会在上传前再次从 HF 下载最新数据并合并，确保所有人的数据都被保留。

### Q3: outputs/ 目录要提交吗？
**A**: 不需要。`outputs/` 已在 `.gitignore` 中，只提交 `hf_data/`。

### Q4: 推送后多久能在 HF 看到结果？
**A**: GitHub Actions 通常在 1-3 分钟内完成，完成后即可在 HuggingFace 查看。

### Q5: 如何查看自动化流程的执行状态？
**A**: 访问 GitHub 仓库的 **Actions** 标签页，查看 "Upload to Hugging Face" 工作流。

### Q6: Actions 被账单限制阻塞时怎么办？
**A**: 先联系仓库管理员恢复 Actions 账单/配额；在恢复前，执行 `bash scripts/local_ci_fallback.sh` 完成本地等价校验，并在 PR/Issue 中附上本地检查结果。

---

## 📚 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/aggregate_for_hf.py` | 本地聚合脚本 |
| `scripts/merge_and_upload.py` | GitHub Actions 合并脚本 |
| `scripts/upload_to_hf.py` | GitHub Actions 上传脚本 |
| `scripts/local_ci_fallback.sh` | Actions 不可用时的本地 CI 兜底脚本 |
| `.github/workflows/upload-to-hf.yml` | 自动化流程配置 |
| `outputs/` | 本地 benchmark 原始结果（不提交） |
| `hf_data/` | 聚合后的数据（提交后会被自动清理） |

---

## 🎯 总结

**你只需要 3 步**：
1. 运行 benchmark
2. 聚合数据
3. 推送到 GitHub

**剩下的都是自动的**！🚀
