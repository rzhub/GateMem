from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .types import EmbeddingConfig
from bench.retrieval.langchain_faiss import make_langchain_embedder


class LangChainEmbeddingRouter:
    """Router-like adapter backed by LangChain embedding implementations.

    This exposes the same public surface as ``EmbeddingRouter`` so agents that
    require an ``embed_router`` (e.g. A-Mem, ReMem, Mem0 builtin) can operate
    with ``embedding_impl=langchain`` instead of silently degrading or failing.
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._provider = make_langchain_embedder(
            provider=cfg.provider,
            model=cfg.model,
            api_base=cfg.api_base,
            api_key_env=cfg.api_key_env,
            device=cfg.device,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
        )

    def embed_texts(self, texts: List[str]) -> Tuple[np.ndarray, dict]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32), {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_s": 0.0,
            }
        vecs = self._provider.embed_documents(texts)
        emb = np.asarray(vecs, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if self.cfg.normalize and emb.size > 0:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
        return emb, {"input_tokens": None, "output_tokens": None, "total_tokens": None, "latency_s": None}

    def embed_query(self, text: str) -> Tuple[np.ndarray, dict]:
        vec = self._provider.embed_query(text)
        emb = np.asarray(vec, dtype=np.float32)
        if emb.ndim != 1:
            emb = emb.reshape(-1)
        if self.cfg.normalize and emb.size > 0:
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
        return emb, {"input_tokens": None, "output_tokens": None, "total_tokens": None, "latency_s": None}
