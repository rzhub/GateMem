"""Minimal, self-contained Mem0 implementation for this benchmark.

We intentionally vendor only the core algorithmic pieces needed for a Mem0 baseline:

1) Fact extraction prompt + JSON parsing (facts: list[str])
2) Similarity search over stored memory items (vector embeddings)
3) LLM-driven memory update actions: ADD / UPDATE / DELETE / NONE

This module is designed to be:
- faithful to Mem0's extraction/update workflow
- dependency-light (no external vector DB required)
- easy to audit and reproduce
"""

from .prompts import build_update_memory_prompt, build_user_fact_extraction_prompts
from .store import Mem0MemoryItem, Mem0Store

__all__ = [
    "Mem0Store",
    "Mem0MemoryItem",
    "build_user_fact_extraction_prompts",
    "build_update_memory_prompt",
]
