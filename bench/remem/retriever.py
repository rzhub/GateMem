from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .graph import MemoryGraph
from .store import NodeStore
from .types import EvidencePacket, MemoryNode, SearchResult


def stable_id(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.md5(text.encode('utf-8')).hexdigest()[:16]}"


def format_fact_for_embedding(fact: Dict[str, Any]) -> str:
    subj = str(fact.get("subject") or "").strip()
    pred = str(fact.get("predicate") or "").strip()
    obj = str(fact.get("object") or "").strip()
    qualifiers = fact.get("qualifiers") if isinstance(fact.get("qualifiers"), dict) else {}
    qparts = []
    for k in ["record_time", "point_in_time", "start_time", "end_time"]:
        v = qualifiers.get(k)
        if v:
            qparts.append(f"{k}: {v}")
    qtxt = " {" + ", ".join(qparts) + "}" if qparts else ""
    return f"({subj}, {pred}, {obj}){qtxt}"


class ReMemIndex:
    """Hybrid gist/fact/entity/verbatim memory index."""

    def __init__(self, *, embed_router: Optional[Any], logger: Any = None, synonymy_threshold: float = 0.8) -> None:
        self.logger = logger
        self.synonymy_threshold = float(synonymy_threshold)
        self.graph = MemoryGraph()
        self.stores: Dict[str, NodeStore] = {
            "verbatim": NodeStore(embed_router=embed_router, logger=logger),
            "gists": NodeStore(embed_router=embed_router, logger=logger),
            "facts": NodeStore(embed_router=embed_router, logger=logger),
            "entity": NodeStore(embed_router=embed_router, logger=logger),
        }
        self.nodes: Dict[str, MemoryNode] = {}

    def clear(self) -> None:
        self.graph.clear()
        for store in self.stores.values():
            store.clear()
        self.nodes.clear()

    def add_turn(
        self,
        *,
        turn_id: str,
        principal_id: str,
        role: str,
        timestamp: Optional[str],
        verbatim_text: str,
        gists: Sequence[str],
        facts: Sequence[Dict[str, Any]],
        entities: Sequence[str],
        record_refs: Sequence[str],
    ) -> Dict[str, List[str]]:
        created: Dict[str, List[str]] = defaultdict(list)

        meta_base = {
            "turn_id": turn_id,
            "principal_id": principal_id,
            "role": role,
            "timestamp": timestamp,
            "record_refs": list(record_refs or []),
        }

        verbatim_id = stable_id("verbatim", f"{turn_id}|{verbatim_text}")
        verbatim_node = MemoryNode(
            node_id=verbatim_id,
            node_type="verbatim",
            content=verbatim_text,
            source_turn_id=turn_id,
            principal_id=principal_id,
            role=role,
            timestamp=timestamp,
            metadata=dict(meta_base),
        )
        self._add_node(verbatim_node)
        created["verbatim"].append(verbatim_id)

        gist_ids: List[str] = []
        for gist in gists:
            gist_id = stable_id("gists", gist)
            gist_node = MemoryNode(
                node_id=gist_id,
                node_type="gists",
                content=gist,
                source_turn_id=turn_id,
                principal_id=principal_id,
                role=role,
                timestamp=timestamp,
                metadata=dict(meta_base),
            )
            self._add_node(gist_node)
            created["gists"].append(gist_id)
            gist_ids.append(gist_id)
            self.graph.add_edge(verbatim_id, gist_id, relation="supports", weight=1.0)

        fact_ids: List[str] = []
        for fact in facts:
            fact_text = format_fact_for_embedding(fact)
            fact_id = stable_id("facts", fact_text)
            fact_meta = dict(meta_base)
            fact_meta["fact"] = dict(fact)
            fact_node = MemoryNode(
                node_id=fact_id,
                node_type="facts",
                content=fact_text,
                source_turn_id=turn_id,
                principal_id=principal_id,
                role=role,
                timestamp=timestamp,
                metadata=fact_meta,
            )
            self._add_node(fact_node)
            created["facts"].append(fact_id)
            fact_ids.append(fact_id)
            self.graph.add_edge(verbatim_id, fact_id, relation="supports", weight=1.0)
            for gist_id in gist_ids:
                self.graph.add_edge(gist_id, fact_id, relation="grounds", weight=1.0)

        entity_to_id: Dict[str, str] = {}
        all_entities = list(entities or [])
        for fact in facts:
            for key in ("subject", "object"):
                v = str(fact.get(key) or "").strip()
                if v:
                    all_entities.append(v)
        seen_entities = set()
        for entity in all_entities:
            ek = entity.lower().strip()
            if not ek or ek in seen_entities:
                continue
            seen_entities.add(ek)
            entity_id = self._get_or_create_entity(
                entity=entity,
                turn_id=turn_id,
                principal_id=principal_id,
                role=role,
                timestamp=timestamp,
                record_refs=record_refs,
            )
            entity_to_id[entity] = entity_id
            created["entity"].append(entity_id)
            self.graph.add_edge(verbatim_id, entity_id, relation="mentions", weight=0.8)
            for gist_id in gist_ids:
                if entity.lower() in self.nodes[gist_id].content.lower():
                    self.graph.add_edge(gist_id, entity_id, relation="mentions", weight=0.8)

        for fact_id in fact_ids:
            fact = self.nodes[fact_id].metadata.get("fact") or {}
            subj = str(fact.get("subject") or "").strip()
            obj = str(fact.get("object") or "").strip()
            subj_id = entity_to_id.get(subj)
            obj_id = entity_to_id.get(obj)
            if subj_id:
                self.graph.add_edge(fact_id, subj_id, relation="subject", weight=1.0)
            if obj_id:
                self.graph.add_edge(fact_id, obj_id, relation="object", weight=1.0)
            if subj_id and obj_id:
                self.graph.add_edge(subj_id, obj_id, relation="co_occurs", weight=0.6)

        self._link_new_gists(gist_ids)
        return dict(created)

    def _add_node(self, node: MemoryNode) -> None:
        if node.node_id in self.nodes:
            # Keep the earliest insertion but do not duplicate store entries.
            return
        self.nodes[node.node_id] = node
        self.stores[node.node_type].add(node)

    def _get_or_create_entity(
        self,
        *,
        entity: str,
        turn_id: str,
        principal_id: str,
        role: str,
        timestamp: Optional[str],
        record_refs: Sequence[str],
    ) -> str:
        entity_id = stable_id("entity", entity.lower())
        if entity_id in self.nodes:
            return entity_id
        node = MemoryNode(
            node_id=entity_id,
            node_type="entity",
            content=entity,
            source_turn_id=turn_id,
            principal_id=principal_id,
            role=role,
            timestamp=timestamp,
            metadata={
                "turn_id": turn_id,
                "principal_id": principal_id,
                "role": role,
                "timestamp": timestamp,
                "record_refs": list(record_refs or []),
            },
        )
        self._add_node(node)
        return entity_id

    def _link_new_gists(self, gist_ids: Sequence[str]) -> None:
        if not gist_ids:
            return
        gist_store = self.stores["gists"]
        for gist_id in gist_ids:
            for other_id, sim in gist_store.similarity_to_existing(gist_id):
                if sim >= self.synonymy_threshold:
                    self.graph.add_edge(gist_id, other_id, relation="synonym", weight=float(sim))

    def semantic_retrieve(self, query: str, *, top_k: int, exclude_ids: Optional[Set[str]] = None) -> Tuple[List[SearchResult], Dict[str, Any]]:
        out: List[SearchResult] = []
        usage_acc = {"input_tokens": 0.0, "total_tokens": 0.0}
        for entry_type, limit in (("gists", top_k), ("facts", top_k), ("entity", max(3, top_k // 2))):
            rows, usage = self.stores[entry_type].semantic_search(query, top_k=limit, allow_ids=None if not exclude_ids else set(self.stores[entry_type].nodes.keys()) - set(exclude_ids))
            out.extend(rows)
            for key in usage_acc:
                val = usage.get(key)
                if isinstance(val, (int, float)):
                    usage_acc[key] += float(val)
        return self._dedup_rank(out, top_k=top_k), usage_acc

    def lexical_retrieve(self, query: str, *, top_k: int, exclude_ids: Optional[Set[str]] = None) -> List[SearchResult]:
        out: List[SearchResult] = []
        for entry_type, limit in (("gists", top_k), ("facts", top_k), ("verbatim", max(2, top_k // 2)), ("entity", max(2, top_k // 2))):
            allow_ids = None if not exclude_ids else set(self.stores[entry_type].nodes.keys()) - set(exclude_ids)
            out.extend(self.stores[entry_type].lexical_search(query, top_k=limit, allow_ids=allow_ids))
        return self._dedup_rank(out, top_k=top_k)

    def expand_gist(self, gist_id: str, *, query: str, limit: int) -> List[SearchResult]:
        scores: Dict[str, float] = {}
        synonym_neighbors = []
        for nb, edge in self.graph.neighbors(gist_id, limit=limit * 8):
            if nb.startswith("gists-") and edge.relation == "synonym":
                synonym_neighbors.append((nb, float(edge.weight)))
            if nb.startswith("gists-") or nb.startswith("facts-") or nb.startswith("verbatim-"):
                scores[nb] = max(scores.get(nb, 0.0), float(edge.weight))
        # Also expose fact / verbatim contexts from synonymous gists, which is central to REMEM-I.
        for syn_id, syn_weight in synonym_neighbors[: limit * 4]:
            for nb, edge in self.graph.neighbors(syn_id, limit=limit * 6):
                if nb.startswith("facts-") or nb.startswith("verbatim-") or nb.startswith("gists-"):
                    scores[nb] = max(scores.get(nb, 0.0), syn_weight * float(edge.weight))
        return self._rerank_candidates(query=query, scored=scores, limit=limit)

    def expand_entity(self, entity_id: str, *, query: str, limit: int) -> List[SearchResult]:
        scores: Dict[str, float] = {}
        for nb, edge in self.graph.neighbors(entity_id, limit=limit * 8):
            if nb.startswith("facts-") or nb.startswith("gists-") or nb.startswith("verbatim-") or nb.startswith("entity-"):
                scores[nb] = max(scores.get(nb, 0.0), float(edge.weight))
            if nb.startswith("facts-"):
                for nb2, edge2 in self.graph.neighbors(nb, limit=4):
                    if nb2.startswith("verbatim-") or nb2.startswith("gists-"):
                        scores[nb2] = max(scores.get(nb2, 0.0), float(edge.weight) * float(edge2.weight))
        return self._rerank_candidates(query=query, scored=scores, limit=limit)

    def collect_evidence(self, scored_nodes: Dict[str, float], *, top_k: int) -> List[SearchResult]:
        verbatim_scores: Dict[str, List[float]] = defaultdict(list)
        fallback_scores: Dict[str, float] = {}
        for node_id, score in scored_nodes.items():
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if node.node_type == "verbatim":
                verbatim_scores[node_id].append(score)
                continue
            fallback_scores[node_id] = max(fallback_scores.get(node_id, 0.0), score)
            if node.node_type == "gists":
                for nb, edge in self.graph.neighbors(node_id, prefix="verbatim-", limit=4):
                    verbatim_scores[nb].append(score * float(edge.weight))
            elif node.node_type == "facts":
                for nb, edge in self.graph.neighbors(node_id, prefix="verbatim-", limit=4):
                    verbatim_scores[nb].append(score * float(edge.weight) * 0.1)
                for gist_nb, edge in self.graph.neighbors(node_id, prefix="gists-", limit=3):
                    for vb, edge2 in self.graph.neighbors(gist_nb, prefix="verbatim-", limit=3):
                        verbatim_scores[vb].append(score * float(edge.weight) * float(edge2.weight) * 0.1)
            elif node.node_type == "entity":
                for fact_nb, edge in self.graph.neighbors(node_id, prefix="facts-", limit=3):
                    for vb, edge2 in self.graph.neighbors(fact_nb, prefix="verbatim-", limit=3):
                        verbatim_scores[vb].append(score * float(edge.weight) * float(edge2.weight) * 0.1)
                    for gist_nb, edge2 in self.graph.neighbors(fact_nb, prefix="gists-", limit=2):
                        for vb, edge3 in self.graph.neighbors(gist_nb, prefix="verbatim-", limit=2):
                            verbatim_scores[vb].append(score * float(edge.weight) * float(edge2.weight) * float(edge3.weight) * 0.1)

        rows: List[SearchResult] = []
        for vb_id, vals in verbatim_scores.items():
            node = self.nodes.get(vb_id)
            if node is None:
                continue
            score = sum(vals) / max(1, len(vals))
            rows.append(SearchResult(node_id=vb_id, node_type=node.node_type, content=node.content, score=score, metadata=node.metadata))
        rows.sort(key=lambda x: (x.score, str(x.metadata.get("timestamp") or "")), reverse=True)
        if rows:
            return rows[:top_k]
        return self._ids_to_results(fallback_scores, limit=top_k)

    def collect_evidence_packets(
        self,
        scored_nodes: Dict[str, float],
        *,
        top_k: int,
        max_gists_per_packet: int = 2,
        max_facts_per_packet: int = 2,
        max_entities_per_packet: int = 2,
    ) -> List[EvidencePacket]:
        verbatim_rows = self.collect_evidence(scored_nodes, top_k=top_k)
        packets: List[EvidencePacket] = []
        for vb in verbatim_rows:
            supporting = self._collect_supporting_rows(
                vb.node_id,
                scored_nodes=scored_nodes,
                max_gists=max_gists_per_packet,
                max_facts=max_facts_per_packet,
                max_entities=max_entities_per_packet,
            )
            packets.append(
                EvidencePacket(
                    verbatim=vb,
                    supporting_gists=supporting["gists"],
                    supporting_facts=supporting["facts"],
                    supporting_entities=supporting["entity"],
                )
            )
        return packets

    def _collect_supporting_rows(
        self,
        verbatim_id: str,
        *,
        scored_nodes: Dict[str, float],
        max_gists: int,
        max_facts: int,
        max_entities: int,
    ) -> Dict[str, List[SearchResult]]:
        candidates: Dict[str, Dict[str, float]] = {"gists": {}, "facts": {}, "entity": {}}

        def _consider(node_id: str, relation_weight: float) -> None:
            node = self.nodes.get(node_id)
            if node is None:
                return
            ntype = node.node_type
            if ntype not in candidates:
                return
            base = float(scored_nodes.get(node_id, 0.0))
            support = max(base, max(0.0, float(relation_weight)) * 0.5)
            prev = candidates[ntype].get(node_id, 0.0)
            if support > prev:
                candidates[ntype][node_id] = support

        for nb, edge in self.graph.neighbors(verbatim_id, limit=12):
            _consider(nb, float(edge.weight))
            if nb.startswith("facts-"):
                for nb2, edge2 in self.graph.neighbors(nb, limit=8):
                    _consider(nb2, float(edge.weight) * float(edge2.weight))
            elif nb.startswith("gists-"):
                for nb2, edge2 in self.graph.neighbors(nb, limit=8):
                    if nb2.startswith("entity-"):
                        _consider(nb2, float(edge.weight) * float(edge2.weight))
        rows = {
            "gists": self._ids_to_results(candidates["gists"], limit=max_gists),
            "facts": self._ids_to_results(candidates["facts"], limit=max_facts),
            "entity": self._ids_to_results(candidates["entity"], limit=max_entities),
        }
        return rows

    def _rerank_candidates(self, *, query: str, scored: Dict[str, float], limit: int) -> List[SearchResult]:
        if not scored:
            return []
        base_by_id = {k: float(v) for k, v in scored.items()}
        merged: Dict[str, SearchResult] = {}
        remaining = set(scored.keys())
        for entry_type in ("gists", "facts", "verbatim", "entity"):
            allow = {cid for cid in remaining if cid.startswith(f"{entry_type}-")}
            if not allow:
                continue
            rows, _ = self.stores[entry_type].semantic_search(query, top_k=max(limit * 4, len(allow)), allow_ids=allow)
            if not rows:
                rows = self.stores[entry_type].lexical_search(query, top_k=max(limit * 4, len(allow)), allow_ids=allow)
            for r in rows:
                boosted = max(base_by_id.get(r.node_id, 0.0), float(r.score))
                merged[r.node_id] = SearchResult(node_id=r.node_id, node_type=r.node_type, content=r.content, score=boosted, metadata=r.metadata)
                remaining.discard(r.node_id)
        # add anything not recovered by search using original graph score
        for node_id in remaining:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            merged[node_id] = SearchResult(node_id=node_id, node_type=node.node_type, content=node.content, score=base_by_id[node_id], metadata=node.metadata)
        out = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return out[:limit]

    def _ids_to_results(self, scored: Dict[str, float], *, limit: int) -> List[SearchResult]:
        rows: List[SearchResult] = []
        for node_id, score in sorted(scored.items(), key=lambda x: x[1], reverse=True):
            node = self.nodes.get(node_id)
            if node is None:
                continue
            rows.append(SearchResult(node_id=node_id, node_type=node.node_type, content=node.content, score=float(score), metadata=node.metadata))
            if len(rows) >= limit:
                break
        return rows

    @staticmethod
    def _dedup_rank(rows: Iterable[SearchResult], *, top_k: int) -> List[SearchResult]:
        by_id: Dict[str, SearchResult] = {}
        for r in rows:
            prev = by_id.get(r.node_id)
            if prev is None or r.score > prev.score:
                by_id[r.node_id] = r
        out = sorted(by_id.values(), key=lambda x: x.score, reverse=True)
        return out[:top_k]
