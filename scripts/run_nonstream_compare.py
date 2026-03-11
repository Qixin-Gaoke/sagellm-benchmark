#!/usr/bin/env python3
"""Thin wrapper for the reusable non-stream compare runner."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _main() -> int:
    from sagellm_benchmark.nonstream_compare import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
