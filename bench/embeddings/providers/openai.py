from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

from ..types import EmbeddingConfig


class OpenAIEmbeddingProvider:
    """OpenAI (or OpenAI-compatible) embeddings provider.

    Uses the /v1/embeddings endpoint and returns L2-normalized float32 vectors
    when ``cfg.normalize`` is enabled.
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.api_base = (cfg.api_base or "https://api.openai.com").rstrip("/")
        self.api_key_env = cfg.api_key_env or "OPENAI_API_KEY"

    def _headers(self) -> dict:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Missing API key in env var {self.api_key_env}. Set it to call OpenAI embeddings."
            )
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _usage_out(self, usage: Dict[str, Any], latency_s: float) -> Dict[str, Any]:
        return {
            "input_tokens": usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("promptTokenCount"),
            "output_tokens": usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("total_tokens") or usage.get("totalTokenCount"),
            "latency_s": latency_s,
        }

    @staticmethod
    def _accum_usage(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            v = src.get(k)
            if isinstance(v, (int, float)):
                prev = dst.get(k)
                dst[k] = float(v) + (float(prev) if isinstance(prev, (int, float)) else 0.0)
        v = src.get("latency_s")
        if isinstance(v, (int, float)):
            prev = dst.get("latency_s")
            dst["latency_s"] = float(v) + (float(prev) if isinstance(prev, (int, float)) else 0.0)

    def _request_batch(self, texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        url = f"{self.api_base}/v1/embeddings"
        payload = {"model": self.cfg.model, "input": texts}

        last_err = None
        for attempt in range(self.cfg.max_retries):
            t0 = time.perf_counter()
            try:
                r = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.cfg.timeout_s,
                )
                if r.status_code >= 400:
                    raise RuntimeError(f"Embedding API error {r.status_code}: {r.text[:500]}")
                data = r.json()
                rows = data.get("data")
                if not isinstance(rows, list) or not rows:
                    raise RuntimeError("Embedding API returned no vectors")
                if len(rows) != len(texts):
                    raise RuntimeError(
                        f"Embedding API returned {len(rows)} vectors for {len(texts)} inputs"
                    )

                ordered: List[Any] = [None] * len(texts)
                for pos, item in enumerate(rows):
                    if not isinstance(item, dict) or "embedding" not in item:
                        raise RuntimeError("Embedding API returned malformed item without `embedding`")
                    idx = item.get("index", pos)
                    if not isinstance(idx, int) or not (0 <= idx < len(texts)):
                        raise RuntimeError(f"Embedding API returned invalid index: {idx!r}")
                    if ordered[idx] is not None:
                        raise RuntimeError(f"Embedding API returned duplicate embedding index: {idx}")
                    ordered[idx] = item["embedding"]
                if any(v is None for v in ordered):
                    raise RuntimeError("Embedding API response omitted one or more requested indices")

                emb = np.asarray(ordered, dtype=np.float32)
                usage_out = self._usage_out(data.get("usage", {}) or {}, time.perf_counter() - t0)
                return emb, usage_out
            except Exception as e:
                last_err = e
                if attempt + 1 < self.cfg.max_retries:
                    time.sleep(0.5 * (2**attempt))
                else:
                    break

        raise RuntimeError(f"Embedding request failed after retries: {last_err}")

    def embed_texts(self, texts: List[str]) -> Tuple[np.ndarray, dict]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32), {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_s": 0.0,
            }

        bs = max(1, int(self.cfg.batch_size))
        vec_batches: List[np.ndarray] = []
        usage_total: Dict[str, Any] = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "latency_s": 0.0,
        }
        for i in range(0, len(texts), bs):
            emb, usage = self._request_batch(texts[i : i + bs])
            vec_batches.append(emb)
            self._accum_usage(usage_total, usage)

        emb = np.concatenate(vec_batches, axis=0) if vec_batches else np.zeros((0, 0), dtype=np.float32)
        if emb.shape[0] != len(texts):
            raise RuntimeError(
                f"Embedding provider produced {emb.shape[0]} vectors for {len(texts)} inputs"
            )
        if self.cfg.normalize and emb.size > 0:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
        return emb, usage_total
