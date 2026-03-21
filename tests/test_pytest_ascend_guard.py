"""Regression tests for pytest-level Ascend autoload guard."""

from __future__ import annotations

import os


def test_pytest_sets_torch_backend_autoload_guard() -> None:
    assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
