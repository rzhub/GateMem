from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .types import EmbeddingConfig


class EmbeddingRouter:
    """Thin router over embedding providers.

    Public API:
      embed_texts(texts) -> (embeddings, usage)

    where embeddings is a float32 numpy array of shape (n, d) and usage is a dict.
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        provider = (cfg.provider or "").lower()
        if provider == "openai":
            from .providers.openai import OpenAIEmbeddingProvider

            self._provider = OpenAIEmbeddingProvider(cfg)
        elif provider == "hf":
            from .providers.hf import HFEmbeddingProvider

            self._provider = HFEmbeddingProvider(cfg)
        else:
            raise ValueError(f"Unknown embedding provider: {cfg.provider}")

    def embed_texts(self, texts: List[str]) -> Tuple[np.ndarray, dict]:
        return self._provider.embed_texts(texts)

    def embed_query(self, text: str) -> Tuple[np.ndarray, dict]:
        emb, usage = self.embed_texts([text])
        if emb.ndim == 1:
            return emb, usage
        if emb.shape[0] < 1:
            raise RuntimeError("Embedding provider returned no vectors for a single-query request")
        return emb[0], usage
