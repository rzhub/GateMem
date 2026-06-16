from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from mem0.vector_stores.base import VectorStoreBase


@dataclass
class _Hit:
    id: str
    payload: Dict[str, Any]
    score: float


class InMemory(VectorStoreBase):
    """A tiny in-memory vector store implementing Mem0's VectorStoreBase.

    This is an adapter to make the upstream Mem0 Memory() runnable without
    external vector DB services in benchmark settings.
    """

    def __init__(
        self,
        *,
        collection_name: str = "mem0",
        distance_strategy: str = "cosine",
        embedding_model_dims: int = 1536,
        **kwargs,
    ) -> None:
        self.collection_name = collection_name
        self.distance_strategy = distance_strategy
        self.embedding_model_dims = int(embedding_model_dims)
        self._ids: List[str] = []
        self._vecs: Optional[np.ndarray] = None  # shape (N, d)
        self._payloads: List[Dict[str, Any]] = []

    # ------------------ collection ops ------------------

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = int(vector_size)
        self.distance_strategy = distance or self.distance_strategy
        self.reset()

    def list_cols(self):
        return [self.collection_name]

    def delete_col(self):
        self.reset()

    def col_info(self):
        return {
            "collection_name": self.collection_name,
            "size": len(self._ids),
            "embedding_model_dims": self.embedding_model_dims,
            "distance_strategy": self.distance_strategy,
        }

    def reset(self):
        self._ids = []
        self._vecs = None
        self._payloads = []

    # ------------------ CRUD ------------------

    def insert(self, vectors, payloads=None, ids=None):
        if ids is None:
            raise ValueError("ids must be provided")
        if payloads is None:
            payloads = [{} for _ in ids]
        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("vectors/payloads/ids length mismatch")

        vec_arr = np.asarray(vectors, dtype=np.float32)
        if vec_arr.ndim == 1:
            vec_arr = vec_arr.reshape(1, -1)
        if vec_arr.shape[1] != self.embedding_model_dims:
            # Allow mem0 to pass different dims; update dims if empty store
            if len(self._ids) == 0:
                self.embedding_model_dims = int(vec_arr.shape[1])
            else:
                raise ValueError(
                    f"Embedding dims mismatch: got {vec_arr.shape[1]} expected {self.embedding_model_dims}"
                )

        for vid, v, p in zip(ids, vec_arr, payloads):
            self._ids.append(str(vid))
            self._payloads.append(dict(p) if isinstance(p, dict) else {})
            if self._vecs is None:
                self._vecs = v.reshape(1, -1)
            else:
                self._vecs = np.vstack([self._vecs, v.reshape(1, -1)])

    def delete(self, vector_id):
        vector_id = str(vector_id)
        if vector_id not in self._ids:
            return
        idx = self._ids.index(vector_id)
        self._ids.pop(idx)
        self._payloads.pop(idx)
        if self._vecs is not None:
            self._vecs = np.delete(self._vecs, idx, axis=0)
            if self._vecs.size == 0:
                self._vecs = None

    def update(self, vector_id, vector=None, payload=None):
        vector_id = str(vector_id)
        if vector_id not in self._ids:
            raise KeyError(vector_id)
        idx = self._ids.index(vector_id)
        if payload is not None:
            if not isinstance(payload, dict):
                raise ValueError("payload must be dict")
            self._payloads[idx] = payload
        if vector is not None:
            v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            if self._vecs is None:
                self._vecs = v
            else:
                self._vecs[idx : idx + 1, :] = v

    def get(self, vector_id):
        vector_id = str(vector_id)
        if vector_id not in self._ids:
            return None
        idx = self._ids.index(vector_id)
        return _Hit(id=self._ids[idx], payload=self._payloads[idx], score=1.0)

    def list(self, filters=None, limit=None):
        hits = []
        for vid, p in zip(self._ids, self._payloads):
            if self._passes_filters(p, filters):
                hits.append(_Hit(id=vid, payload=p, score=1.0))
                if limit is not None and len(hits) >= int(limit):
                    break
        return hits

    # ------------------ search ------------------

    def search(self, query, vectors, limit=5, filters=None):
        if self._vecs is None or len(self._ids) == 0:
            return []

        q = np.asarray(vectors, dtype=np.float32).reshape(1, -1)
        M = self._vecs

        # filter candidates
        idxs = [i for i, p in enumerate(self._payloads) if self._passes_filters(p, filters)]
        if not idxs:
            return []
        cand = M[idxs]

        if self.distance_strategy == "inner_product":
            scores = (cand @ q.T).reshape(-1)
        else:
            # cosine
            cand_norm = np.linalg.norm(cand, axis=1) + 1e-8
            q_norm = float(np.linalg.norm(q) + 1e-8)
            scores = ((cand @ q.T).reshape(-1)) / (cand_norm * q_norm)

        # take top-k
        k = max(1, int(limit))
        top_idx_local = np.argsort(-scores)[:k]
        out: List[_Hit] = []
        for j in top_idx_local:
            i = idxs[int(j)]
            out.append(_Hit(id=self._ids[i], payload=self._payloads[i], score=float(scores[int(j)])))
        return out

    # ------------------ helpers ------------------

    @staticmethod
    def _passes_filters(payload: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        if not isinstance(filters, dict):
            return True
        for k, v in filters.items():
            # Only support exact-match filters for benchmark.
            if isinstance(v, dict):
                # If advanced operators are provided, fall back to non-match.
                # (Mem0 Memory() pre-processes advanced filters; our benchmark doesn't rely on them.)
                return False
            if payload.get(k) != v:
                return False
        return True
