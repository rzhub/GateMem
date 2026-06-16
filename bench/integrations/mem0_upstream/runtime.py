from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional


def _install_module(name: str, module: ModuleType) -> None:
    # Only set if missing; respect user's installed packages.
    if name not in sys.modules:
        sys.modules[name] = module


def ensure_upstream_ready(*, repo_root: Optional[str] = None, state_dir: Optional[str] = None) -> str:
    """Make vendored Mem0 upstream importable and install shims as needed.

    Returns the upstream_root path that was added to sys.path.
    """

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    upstream_root = root / "third_party" / "mem0_upstream"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    # Install shims only if user doesn't already have these packages.
    openai_shim = importlib.import_module("bench.integrations.mem0_upstream.shims.openai_shim")
    posthog_stub = importlib.import_module("bench.integrations.mem0_upstream.shims.posthog_stub")
    _install_module("openai", openai_shim)
    _install_module("posthog", posthog_stub)

    # Optional: direct Mem0's state dir (history DB default) to a controlled location.
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
        os.environ["MEM0_DIR"] = state_dir

    return str(upstream_root)
