from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Retrieved:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class EmbeddingRetriever:
    """Embedding-based cosine retriever ("real RAG").

    - Uses an EmbeddingRouter to generate embeddings for chunks and queries.
    - Uses brute-force cosine similarity by default.
    - If `faiss` is installed, you can enable `use_faiss=True`.

    Notes:
      - We assume embeddings are L2-normalized (router can normalize).
      - For small datasets (<= few thousand chunks), brute force is fine.
    """

    def __init__(
        self,
        *,
        embed_router: Any,
        use_faiss: bool = False,
        faiss_index_factory: str = "Flat",
        show_progress: bool = False,
        logger: Any = None,
    ) -> None:
        self.embed_router = embed_router
        self.use_faiss = use_faiss
        self.faiss_index_factory = faiss_index_factory
        self.show_progress = show_progress
        self.logger = logger

        self._chunks: List[Tuple[str, str, Dict[str, Any]]] = []
        self._emb: Optional[np.ndarray] = None

        self._faiss = None
        self._faiss_index = None

        if self.use_faiss:
            try:
                import faiss  # type: ignore

                self._faiss = faiss
            except Exception as e:
                raise RuntimeError(
                    "FAISS not installed. Install faiss-cpu to use use_faiss=True."
                ) from e

    def add(self, chunk_id: str, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Embed and add a chunk."""
        if self.logger:
            self.logger.debug("Embedding add: chunk_id=%s chars=%d", chunk_id, len(text))
        emb, usage = self.embed_router.embed_texts([text])
        vec = emb.astype(np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        self._chunks.append((chunk_id, text, metadata))

        if self._emb is None:
            self._emb = vec
        else:
            self._emb = np.concatenate([self._emb, vec], axis=0)

        if self._faiss is not None:
            if self._faiss_index is None:
                d = self._emb.shape[1]
                if self.faiss_index_factory.lower() == "flat":
                    self._faiss_index = self._faiss.IndexFlatIP(d)  # cosine if vectors normalized
                else:
                    # fallback to flat
                    self._faiss_index = self._faiss.IndexFlatIP(d)
                self._faiss_index.add(vec)
            else:
                self._faiss_index.add(vec)

        return usage

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        allow_chunk_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[Retrieved], float, Dict[str, Any]]:
        t0 = time.perf_counter()
        if not self._chunks or self._emb is None:
            return [], time.perf_counter() - t0, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        qv, usage = self.embed_router.embed_query(query)
        qv = qv.astype(np.float32)

        if self.logger:
            self.logger.debug("Embedding search: top_k=%d allow_filter=%s", top_k, bool(allow_chunk_ids))

        if self._faiss_index is not None:
            # FAISS returns inner products
            D, I = self._faiss_index.search(qv.reshape(1, -1), min(top_k * 5, len(self._chunks)))
            candidates = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]
        else:
            # brute force cosine = dot product if normalized
            sims = (self._emb @ qv.reshape(-1, 1)).reshape(-1)
            idxs = np.argsort(-sims)
            candidates = [(int(i), float(sims[i])) for i in idxs[: min(top_k * 5, len(idxs))]]

        out: List[Retrieved] = []
        for i, s in candidates:
            cid, text, meta = self._chunks[i]
            if allow_chunk_ids is not None and cid not in allow_chunk_ids:
                continue
            if s <= 0:
                continue
            out.append(Retrieved(chunk_id=cid, score=s, text=text, metadata=meta))
            if len(out) >= top_k:
                break

        return out, time.perf_counter() - t0, usage

    @property
    def size(self) -> int:
        return len(self._chunks)
