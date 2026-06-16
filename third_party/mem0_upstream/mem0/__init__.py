"""Vendored Mem0 upstream package.

This benchmark vendors Mem0 upstream source for reproducibility without requiring
pip installation. Upstream expects the distribution name `mem0ai`; when vendored
in-tree, package metadata is not available, so we fall back to a placeholder
version string.

We intentionally avoid importing optional client helpers at import time to reduce
extra dependencies; baselines only require `mem0.memory.main.Memory`.
"""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("mem0ai")
except Exception:
    __version__ = "vendored"

from mem0.memory.main import AsyncMemory, Memory  # noqa
