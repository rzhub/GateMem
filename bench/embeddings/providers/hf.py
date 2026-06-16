from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from ..types import EmbeddingConfig


class HFEmbeddingProvider:
    """HuggingFace embedding provider.

    Preferred path:
    - ``sentence_transformers.SentenceTransformer`` when available, which gives
      stronger default behavior for common embedding checkpoints.

    Fallback path:
    - ``transformers.AutoTokenizer`` + ``AutoModel`` with attention-mask-aware
      mean pooling.
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._backend = "transformers"
        self._st_model = None
        self.max_length = max(1, int(cfg.max_length)) if int(cfg.max_length) > 0 else 512

        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "HF embedding provider requires `torch`. Install with: pip install torch"
            ) from e
        self.torch = torch
        self.device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(cfg.model, device=self.device)
            try:
                model.max_seq_length = self.max_length
            except Exception:
                pass
            self._backend = "sentence_transformers"
            self._st_model = model
            return
        except Exception:
            self._st_model = None

        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "HF embedding provider requires `transformers` (or `sentence-transformers`). "
                "Install with: pip install transformers"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModel.from_pretrained(cfg.model)
        self.model.eval()
        if self.device == "cuda":
            self.model.to("cuda")

    def _mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def embed_texts(self, texts: List[str]) -> Tuple[np.ndarray, dict]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32), {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "latency_s": 0.0,
            }

        t0 = time.perf_counter()
        bs = max(1, int(self.cfg.batch_size))

        if self._backend == "sentence_transformers" and self._st_model is not None:
            emb = self._st_model.encode(
                texts,
                batch_size=bs,
                convert_to_numpy=True,
                normalize_embeddings=bool(self.cfg.normalize),
                show_progress_bar=False,
            )
            emb = np.asarray(emb, dtype=np.float32)
        else:
            vecs = []
            with self.torch.no_grad():
                for i in range(0, len(texts), bs):
                    batch = texts[i : i + bs]
                    enc = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=self.max_length,
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    out = self.model(**enc)
                    pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                    vecs.append(pooled.detach().cpu().numpy().astype(np.float32))
            emb = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 0), dtype=np.float32)
            if self.cfg.normalize and emb.size > 0:
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb = emb / norms

        usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "latency_s": time.perf_counter() - t0,
        }
        return emb, usage
