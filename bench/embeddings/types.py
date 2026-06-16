from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding providers.

    Notes:
      - provider: 'openai' or 'hf'
      - api_base: for OpenAI-compatible servers; default OpenAI public endpoint
      - api_key_env: environment variable name that stores the API key
    """

    provider: str = "openai"  # openai|hf
    model: str = "text-embedding-3-small"

    # OpenAI / OpenAI-compatible
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None

    timeout_s: float = 60.0
    max_retries: int = 3

    # HF local models (optional dependency)
    device: str = "cpu"  # cpu|cuda
    batch_size: int = 16
    max_length: int = 512
    normalize: bool = True
