# Repo Size Policy (Issue #13)

## Problem

Large data files (model weights, raw datasets, benchmark artifacts) committed to
the Git repository make it slow to clone for developers with limited bandwidth.

## Policy (Mandatory)

### ✅ Allowed in Git

| Type | Path | Reason |
|------|------|--------|
| Source code | `src/` | Small, text-only |
| Tests | `tests/` | Small, text-only |
| Small fixtures | `tests/fixtures/*.json` | < 50 KB each |
| Leaderboard JSON | `hf_data/*.json` | Auto-generated, < 1 MB |
| Documentation | `docs/*.md` | Text |
| Configuration | `pyproject.toml`, `quickstart.sh`, `.github/` | Text |

### ❌ Forbidden in Git

| Type | Reason | Where Instead |
|------|--------|---------------|
| Model weights (`*.safetensors`, `*.bin`, `*.pt`) | GBs per model | HuggingFace Hub |
| Raw datasets (`*.parquet`, `*.arrow`, `*.jsonl.gz`) | Can be 100s of MB | HuggingFace Datasets |
| Benchmark output archives (`outputs/**/*.jsonl`) | Can grow unboundedly | Local only / object storage |
| HuggingFace cache (`hf_datasets_cache/`) | Auto-generated | `~/.cache` (local) |
| ONNX / GGUF model files | Large binaries | HuggingFace Hub |

## How to Use Large Datasets

All ShareGPT / TPCH datasets are loaded at runtime via HuggingFace `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", ...)
```

Set `HF_TOKEN` in your `.env` file if accessing private datasets.

## Benchmark Results Storage

Run benchmark → results are saved to `outputs/` (gitignored) → aggregate with
the CLI → upload to HuggingFace Datasets:

```bash
# 1. Run benchmark
sagellm-benchmark run --model Qwen2-7B --workload all

# 2. Aggregate results locally
sagellm-benchmark aggregate

# 3. Upload to HuggingFace (requires HF_TOKEN)
sagellm-benchmark upload --dataset intellistream/sagellm-benchmark-results
```

The `hf_data/` directory is committed because it contains only the final
aggregated leaderboard JSON (< 1 MB), not raw benchmark data.

## Commit History Note

If the repository has large files in its Git history (from before this policy
was enforced), they can be removed using:

```bash
# Option 1: git-filter-repo (recommended)
pip install git-filter-repo
git filter-repo --path data/raw/ --invert-paths
git filter-repo --path hf_datasets_cache/ --invert-paths

# After rewriting history, force-push (coordinate with team)
git push --force-with-lease origin main
```

⚠️ **Rewriting history breaks all existing forks/clones**. Coordinate with the
team before doing this on the main branch.

## Enforcement

The `.gitignore` blocks accidental commits of large files. Additionally, the
pre-commit hook runs `ruff format` and `ruff check` but does NOT scan for large
files. Developers are responsible for not `git add`-ing data files.

Future improvement: add a `check-added-large-files` pre-commit hook from
<https://pre-commit.com/#check-added-large-files>.
