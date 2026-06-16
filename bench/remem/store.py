from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from bench.retrieval.tfidf import TfidfRetriever

from .types import MemoryNode, SearchResult


@dataclass
class StoreUsage:
    input_tokens: float = 0.0
    total_tokens: float = 0.0

    def add(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        it = usage.get("input_tokens")
        tt = usage.get("total_tokens")
        if isinstance(it, (int, float)):
            self.input_tokens += float(it)
        if isinstance(tt, (int, float)):
            self.total_tokens += float(tt)


class NodeStore:
    """Per-type content store with semantic and lexical retrieval."""

    def __init__(self, *, embed_router: Optional[Any] = None, logger: Any = None) -> None:
        self.embed_router = embed_router
        self.logger = logger
        self.lexical = TfidfRetriever(logger=logger)
        self.nodes: Dict[str, MemoryNode] = {}
        self._id_list: List[str] = []
        self._emb: Optional[np.ndarray] = None
        self.usage = StoreUsage()

    def clear(self) -> None:
        self.lexical = TfidfRetriever(logger=self.logger)
        self.nodes = {}
        self._id_list = []
        self._emb = None
        self.usage = StoreUsage()

    def add(self, node: MemoryNode) -> None:
        self.nodes[node.node_id] = node
        self._id_list.append(node.node_id)
        self.lexical.add(node.node_id, node.content, node.metadata)
        if self.embed_router is not None:
            emb, usage = self.embed_router.embed_texts([node.content])
            vec = emb.astype(np.float32)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)
            if self._emb is None:
                self._emb = vec
            else:
                self._emb = np.concatenate([self._emb, vec], axis=0)
            self.usage.add(usage)

    def get(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    def semantic_search(
        self,
        query: str,
        *,
        top_k: int,
        allow_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        if not self.nodes:
            return [], {"input_tokens": 0.0, "total_tokens": 0.0}
        if self.embed_router is None or self._emb is None:
            return self.lexical_search(query, top_k=top_k, allow_ids=allow_ids), {
                "input_tokens": 0.0,
                "total_tokens": 0.0,
            }

        qv, usage = self.embed_router.embed_query(query)
        sims = (self._emb @ qv.reshape(-1, 1)).reshape(-1)
        idxs = np.argsort(-sims)
        out: List[SearchResult] = []
        for i in idxs:
            score = float(sims[int(i)])
            if score <= 0:
                break
            node_id = self._id_list[int(i)]
            if allow_ids is not None and node_id not in allow_ids:
                continue
            node = self.nodes[node_id]
            out.append(
                SearchResult(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    content=node.content,
                    score=score,
                    metadata=node.metadata,
                )
            )
            if len(out) >= top_k:
                break
        return out, usage

    def lexical_search(
        self,
        query: str,
        *,
        top_k: int,
        allow_ids: Optional[Set[str]] = None,
    ) -> List[SearchResult]:
        rows, _ = self.lexical.search(query, top_k=top_k, allow_chunk_ids=allow_ids)
        out: List[SearchResult] = []
        for r in rows:
            node = self.nodes.get(r.chunk_id)
            if node is None:
                continue
            out.append(
                SearchResult(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    content=node.content,
                    score=float(r.score),
                    metadata=node.metadata,
                )
            )
        return out

    def similarity_to_existing(self, node_id: str) -> List[Tuple[str, float]]:
        if self._emb is None or self.embed_router is None or node_id not in self.nodes:
            return []
        idx = self._id_list.index(node_id)
        vec = self._emb[idx]
        sims = (self._emb @ vec.reshape(-1, 1)).reshape(-1)
        out: List[Tuple[str, float]] = []
        for i, score in enumerate(sims):
            if i == idx:
                continue
            if not math.isfinite(float(score)):
                continue
            out.append((self._id_list[i], float(score)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
