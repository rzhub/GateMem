from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Mem0MemoryItem:
    """A single Mem0 memory entry."""

    id: str
    text: str
    embedding: np.ndarray  # shape (d,)


def _cosine_sim_matrix(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of vectors."""
    q = query.astype(np.float32)
    m = mat.astype(np.float32)
    qn = np.linalg.norm(q) + 1e-12
    mn = np.linalg.norm(m, axis=1) + 1e-12
    return (m @ q) / (mn * qn)


class Mem0Store:
    """A simple in-memory vector store for Mem0."""

    def __init__(self) -> None:
        self._items: Dict[str, Mem0MemoryItem] = {}
        self._next_int_id: int = 0

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[Mem0MemoryItem]:
        return list(self._items.values())

    def get(self, item_id: str) -> Optional[Mem0MemoryItem]:
        return self._items.get(item_id)

    def _alloc_id(self) -> str:
        while True:
            cand = str(self._next_int_id)
            self._next_int_id += 1
            if cand not in self._items:
                return cand

    def add(self, *, text: str, embedding: np.ndarray, suggested_id: Optional[str] = None) -> str:
        item_id = (suggested_id or "").strip()
        if not item_id or item_id in self._items:
            item_id = self._alloc_id()
        self._items[item_id] = Mem0MemoryItem(id=item_id, text=text, embedding=embedding.astype(np.float32))
        return item_id

    def update(self, *, item_id: str, text: str, embedding: np.ndarray) -> bool:
        if item_id not in self._items:
            return False
        self._items[item_id] = Mem0MemoryItem(id=item_id, text=text, embedding=embedding.astype(np.float32))
        return True

    def delete(self, *, item_id: str) -> bool:
        if item_id not in self._items:
            return False
        del self._items[item_id]
        return True

    def search(self, *, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Mem0MemoryItem, float]]:
        if not self._items:
            return []
        items = list(self._items.values())
        mat = np.stack([it.embedding for it in items], axis=0)
        sims = _cosine_sim_matrix(query_embedding, mat)
        k = min(max(1, int(top_k)), len(items))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        # sort by score desc
        idx = idx[np.argsort(-sims[idx])]
        return [(items[i], float(sims[i])) for i in idx]
