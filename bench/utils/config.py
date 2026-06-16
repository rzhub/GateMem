from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class LoadedConfig:
    path: str
    data: Dict[str, Any]


def _expand_env(obj: Any) -> Any:
    """Recursively expand ${VAR} in strings."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def load_config(path: str) -> LoadedConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".json"}:
        data = json.loads(text)
    elif p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
        data = yaml.safe_load(text) or {}
    else:
        raise ValueError("Config must be .json or .yaml/.yml")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict")

    data = _expand_env(data)
    return LoadedConfig(path=str(p), data=data)


def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested config sections into argparse-style keys.

    Supported forms:
      {"llm": {"provider": "openai"}} -> {"llm_provider": "openai"}
      {"judge": {"model": "gpt-4o-mini"}} -> {"judge_model": "gpt-4o-mini"}

    Unknown nested dicts are passed through as-is.
    """

    out: Dict[str, Any] = {}

    def merge(prefix: str, d: Dict[str, Any]):
        for k, v in d.items():
            out[f"{prefix}{k}"] = v

    for k, v in cfg.items():
        if k in {"llm", "judge", "embed", "mem0", "retrieval", "a_mem"} and isinstance(v, dict):
            # Map section names to argparse prefixes
            prefix_map = {
                "llm": "llm_",
                "judge": "judge_",
                "embed": "embed_",
                "mem0": "mem0_",
                "retrieval": "",
                "a_mem": "a_mem_",
            }
            merge(prefix_map[k], v)
        else:
            out[k] = v

    return out
