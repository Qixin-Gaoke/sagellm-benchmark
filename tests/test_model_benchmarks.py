from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sagellm_benchmark.performance.model_benchmarks import _discover_max_seq_len


@pytest.mark.asyncio
async def test_discover_max_seq_len_prefers_local_model_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str):
            del url
            return SimpleNamespace(status_code=404, json=lambda: {})

    class _FakeConfig:
        max_position_embeddings = 32768

    def _from_pretrained(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["hf_endpoint"] = os.environ.get("HF_ENDPOINT")
        return _FakeConfig()

    monkeypatch.setenv("SAGELLM_BENCHMARK_LOCAL_MODEL_DIR", "/tmp/local-model-dir")
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.os.path.exists",
        lambda path: path == "/tmp/local-model-dir",
    )
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=_from_pretrained)),
    )
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    max_seq_len = await _discover_max_seq_len(
        client=object(),
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        backend_url="http://127.0.0.1:9100/v1",
    )

    assert max_seq_len == 32768
    assert captured["args"] == ("/tmp/local-model-dir",)
    assert captured["kwargs"] == {"trust_remote_code": True, "local_files_only": True}
    assert captured["hf_endpoint"] == "https://hf-mirror.com"


@pytest.mark.asyncio
async def test_discover_max_seq_len_remote_path_uses_hf_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str):
            del url
            return SimpleNamespace(status_code=404, json=lambda: {})

    class _FakeConfig:
        n_positions = 4096

    def _from_pretrained(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["hf_endpoint"] = os.environ.get("HF_ENDPOINT")
        return _FakeConfig()

    monkeypatch.delenv("SAGELLM_BENCHMARK_LOCAL_MODEL_DIR", raising=False)
    monkeypatch.delenv("VLLM_LOCAL_MODEL_DIR", raising=False)
    monkeypatch.delenv("HF_LOCAL_MODEL_DIR", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.Path.home", lambda: Path("/no-local-cache")
    )
    monkeypatch.setattr("sagellm_benchmark.clients.openai_client.Path.exists", lambda self: False)
    monkeypatch.setattr(
        "sagellm_benchmark.clients.openai_client.os.path.exists", lambda path: False
    )
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=_from_pretrained)),
    )

    max_seq_len = await _discover_max_seq_len(
        client=object(),
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        backend_url="http://127.0.0.1:9100/v1",
    )

    assert max_seq_len == 4096
    assert captured["args"] == ("Qwen/Qwen2.5-1.5B-Instruct",)
    assert captured["kwargs"] == {"trust_remote_code": True, "local_files_only": False}
    assert captured["hf_endpoint"] == "https://hf-mirror.com"
