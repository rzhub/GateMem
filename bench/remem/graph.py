from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


class MemoryGraph:
    """Small undirected typed graph for ReMem-style episodic retrieval."""

    def __init__(self) -> None:
        self._adj: Dict[str, Dict[str, Edge]] = defaultdict(dict)

    def clear(self) -> None:
        self._adj.clear()

    def add_edge(self, a: str, b: str, relation: str, weight: float = 1.0) -> None:
        if not a or not b or a == b:
            return
        edge = Edge(source=a, target=b, relation=relation, weight=float(weight))
        self._adj[a][b] = edge
        self._adj[b][a] = Edge(source=b, target=a, relation=relation, weight=float(weight))

    def neighbors(
        self,
        node_id: str,
        *,
        relation: Optional[str] = None,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[str, Edge]]:
        items = []
        for nb, edge in self._adj.get(node_id, {}).items():
            if relation is not None and edge.relation != relation:
                continue
            if prefix is not None and not nb.startswith(prefix):
                continue
            items.append((nb, edge))
        items.sort(key=lambda x: x[1].weight, reverse=True)
        if limit is not None:
            return items[: max(0, int(limit))]
        return items

    def multi_hop_neighbors(
        self,
        seeds: Iterable[str],
        *,
        hops: int = 1,
        prefixes: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        frontier = {s: 1.0 for s in seeds if s}
        seen = set(frontier)
        for _ in range(max(0, int(hops))):
            nxt: Dict[str, float] = {}
            for node_id, carry in frontier.items():
                for nb, edge in self._adj.get(node_id, {}).items():
                    if prefixes is not None and not any(nb.startswith(p) for p in prefixes):
                        continue
                    score = carry * max(0.0, float(edge.weight))
                    if score <= 0:
                        continue
                    if score > out.get(nb, 0.0):
                        out[nb] = score
                    if nb not in seen or score > nxt.get(nb, 0.0):
                        nxt[nb] = score
                        seen.add(nb)
            frontier = nxt
            if not frontier:
                break
        return out
