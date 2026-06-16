from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseMemoryAgent, Checkpoint, Turn
from bench.domains import get_domain_label, detect_domain_from_episode
from bench.retrieval import EmbeddingRetriever, TfidfRetriever


@dataclass
class AMemoryNode:
    mem_id: str
    turn_id: str
    principal_id: str
    role: str
    timestamp: Optional[str]
    text: str
    record_refs: List[str] = field(default_factory=list)
    memory_ops: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    linked_ids: List[str] = field(default_factory=list)

    def retrieval_text(self) -> str:
        meta_parts = []
        if self.summary:
            meta_parts.append(f"summary: {self.summary}")
        if self.keywords:
            meta_parts.append("keywords: " + ", ".join(self.keywords))
        if self.entities:
            meta_parts.append("entities: " + ", ".join(self.entities))
        if self.categories:
            meta_parts.append("categories: " + ", ".join(self.categories))
        if self.record_refs:
            meta_parts.append("record_refs: " + ", ".join(self.record_refs))
        meta = " | ".join(meta_parts)
        if meta:
            return (
                f"turn_id={self.turn_id}; principal={self.principal_id}; role={self.role}; "
                f"text={self.text}\n{meta}"
            )
        return f"turn_id={self.turn_id}; principal={self.principal_id}; role={self.role}; text={self.text}"


class AMemAgent(BaseMemoryAgent):
    """A-Mem-style agentic memory baseline (benchmark-adapted).

    Core ideas adapted from the official A-Mem implementation:
    - store each interaction as a memory node
    - attach structured metadata (summary/keywords/entities/categories)
    - evolve memory graph by linking semantically related memories
    - retrieve with semantic search + graph expansion + lightweight rerank

    Notes:
    - `metadata_mode=heuristic` (default) avoids extra LLM calls during ingest and is much cheaper.
    - `metadata_mode=llm` uses the benchmark LLM router to generate structured metadata, closer to A-Mem's
      agentic memory formation, but significantly more expensive.
    """

    def __init__(
        self,
        *,
        top_k: int = 5,
        llm_mode: str = "leaky",
        llm_router: Any = None,
        query_prompt_path: str | None = None,
        retrieval_backend: str = "tfidf",  # tfidf|embedding
        embed_router: Optional[Any] = None,
        use_faiss: bool = False,
        metadata_mode: str = "heuristic",  # heuristic|llm
        link_top_m: int = 3,
        link_score_threshold: float = 0.15,
        graph_expand_hops: int = 1,
        graph_expand_per_hit: int = 2,
        rerank_role_bonus: float = 0.08,
        rerank_entity_bonus: float = 0.06,
        answer_protocol: str = "standard",
        native_include_links: bool = True,
        native_max_links_per_item: int = 2,
    ) -> None:
        super().__init__(
            top_k=top_k,
            llm_mode=llm_mode,
            llm_router=llm_router,
            query_prompt_path=query_prompt_path,
            answer_protocol=answer_protocol,
        )
        if metadata_mode not in {"heuristic", "llm"}:
            raise ValueError("metadata_mode must be heuristic|llm")
        self.retrieval_backend = retrieval_backend
        self._embed_router = embed_router
        self._use_faiss = use_faiss
        self.metadata_mode = metadata_mode
        self.link_top_m = max(0, int(link_top_m))
        self.link_score_threshold = float(link_score_threshold)
        self.graph_expand_hops = max(0, int(graph_expand_hops))
        self.graph_expand_per_hit = max(0, int(graph_expand_per_hit))
        self.rerank_role_bonus = float(rerank_role_bonus)
        self.rerank_entity_bonus = float(rerank_entity_bonus)
        self.native_include_links = bool(native_include_links)
        self.native_max_links_per_item = max(0, int(native_max_links_per_item))

        self._nodes: List[AMemoryNode] = []
        self._nodes_by_id: Dict[str, AMemoryNode] = {}
        self._adj: Dict[str, List[str]] = {}
        self._embed_usage_accum: Dict[str, float] = {"input_tokens": 0.0, "total_tokens": 0.0}
        self._meta_llm_usage_accum: Dict[str, float] = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
        }
        self._domain_label = "general"

        if retrieval_backend == "embedding":
            if embed_router is None:
                raise ValueError("a_mem retrieval_backend=embedding requires embed_router")
            self.retriever = EmbeddingRetriever(
                embed_router=embed_router,
                use_faiss=use_faiss,
                show_progress=False,
                logger=self.logger,
            )
        else:
            self.retriever = TfidfRetriever(logger=self.logger)

    def reset(self, episode: Dict[str, Any]) -> None:
        super().reset(episode)
        self._domain_label = get_domain_label(detect_domain_from_episode(episode))
        self._nodes = []
        self._nodes_by_id = {}
        self._adj = {}
        self._embed_usage_accum = {"input_tokens": 0.0, "total_tokens": 0.0}
        self._meta_llm_usage_accum = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
        if self.retrieval_backend == "embedding":
            self.retriever = EmbeddingRetriever(
                embed_router=self._embed_router,
                use_faiss=self._use_faiss,
                show_progress=False,
                logger=self.logger,
            )
        else:
            self.retriever = TfidfRetriever(logger=self.logger)

    # -------------------- ingest / memory formation --------------------

    def ingest(self, turn: Turn) -> None:
        node = self._build_node_from_turn(turn)
        self._nodes.append(node)
        self._nodes_by_id[node.mem_id] = node
        self._adj.setdefault(node.mem_id, [])

        # semantic indexing (A-Mem retrieval substrate)
        meta = {
            "turn_id": node.turn_id,
            "principal_id": node.principal_id,
            "role": node.role,
            "record_refs": list(node.record_refs),
            "timestamp": node.timestamp,
            "summary": node.summary,
            "keywords": list(node.keywords),
            "entities": list(node.entities),
            "categories": list(node.categories),
        }
        if isinstance(self.retriever, EmbeddingRetriever):
            usage = self.retriever.add(node.mem_id, node.retrieval_text(), meta)
            self._accum_usage(self._embed_usage_accum, usage)
        else:
            self.retriever.add(node.mem_id, node.retrieval_text(), meta)  # type: ignore[arg-type]

        # memory evolution: link to previously related memories
        self._evolve_links(node)

    def _build_node_from_turn(self, turn: Turn) -> AMemoryNode:
        meta = self._analyze_turn_metadata(turn)
        return AMemoryNode(
            mem_id=f"mem_{turn.turn_id}",
            turn_id=turn.turn_id,
            principal_id=turn.speaker_principal_id,
            role=turn.speaker_role,
            timestamp=turn.timestamp,
            text=turn.text,
            record_refs=list(turn.record_refs or []),
            memory_ops=list(turn.memory_ops or []),
            summary=str(meta.get("summary") or "").strip(),
            keywords=self._norm_str_list(meta.get("keywords")),
            entities=self._norm_str_list(meta.get("entities")),
            categories=self._norm_str_list(meta.get("categories")),
        )

    def _analyze_turn_metadata(self, turn: Turn) -> Dict[str, Any]:
        if self.metadata_mode == "llm":
            parsed = self._analyze_turn_metadata_llm(turn)
            if parsed is not None:
                return parsed
            # fallback when LLM parse fails
            self.logger.warning("A-Mem metadata LLM parse failed for %s; falling back to heuristic", turn.turn_id)
        return self._analyze_turn_metadata_heuristic(turn)

    def _analyze_turn_metadata_heuristic(self, turn: Turn) -> Dict[str, Any]:
        text = turn.text or ""
        lower = text.lower()
        # summary: lightly normalized original sentence
        summary = re.sub(r"\s+", " ", text).strip()
        summary = summary[:220]

        # entities: principal names / ids mentioned in text + speaker
        entities = [turn.speaker_principal_id]
        if self.episode:
            for p in ((self.episode.get("entities") or {}).get("principals") or []):
                pid = str(p.get("principal_id", ""))
                dn = str(p.get("display_name", ""))
                if pid and pid != turn.speaker_principal_id:
                    # match any token from pid suffix or display name surface
                    pid_parts = [x for x in re.split(r"[_\W]+", pid) if len(x) >= 4]
                    if any(part.lower() in lower for part in pid_parts):
                        entities.append(pid)
                if dn and dn.lower() in lower:
                    if pid:
                        entities.append(pid)
                    else:
                        entities.append(dn)
        entities = self._dedup_keep_order(entities)

        # categories and keywords (domain-specific heuristics)
        categories: List[str] = []
        if self._domain_label == "education":
            kw_patterns = [
                ("funding", ["stipend", "scholarship", "fellowship", "grant", "aid", "funding", "tuition"]),
                ("academic", ["grade", "course", "syllabus", "exam", "registration", "registrar", "credit"]),
                ("research", ["lab", "proposal", "committee", "irb", "sponsor", "project", "advisor"]),
                ("housing", ["housing", "dorm", "room", "residence", "resident assistant", "res hall"]),
                ("support", ["counselor", "support", "wellness", "accommodation", "conduct"]),
                ("credential", ["token", "credential", "vpn", "access code", "badge", "lab access"]),
                ("privacy", ["restricted", "authorized", "ferpa", "release", "delegate", "parent"]),
                ("deletion", ["delete", "deleted", "remove", "cache", "history"]),
            ]
        elif self._domain_label == "household":
            kw_patterns = [
                ("access", ["door code", "pin", "garage", "alarm", "entry", "unlock", "access window"]),
                ("schedule", ["pickup", "drop-off", "travel", "away", "arrival", "time window", "calendar"]),
                ("caregiving", ["caregiver", "nanny", "elder", "medication", "supervision", "wellness"]),
                ("service", ["cleaner", "cleaning", "maintenance", "technician", "building", "repair"]),
                ("device", ["camera", "smart display", "device", "personal results", "shared tablet"]),
                ("finance", ["bill", "budget", "quote", "refund", "payment", "subscription"]),
                ("privacy", ["private", "authorized", "approved", "guest", "resident", "trusted contact"]),
                ("deletion", ["delete", "deleted", "remove", "revoke", "revoked", "cache", "history"]),
            ]
        elif self._domain_label == "office":
            kw_patterns = [
                ("project", ["project", "program", "stream", "workstream", "roadmap", "milestone", "launch"]),
                ("customer", ["customer", "client", "pilot", "tenant", "account", "partner"]),
                ("finance", ["budget", "invoice", "discount", "pricing", "quote", "procurement", "po", "purchase order"]),
                ("contract", ["contract", "msa", "sow", "dpa", "nda", "legal", "renewal", "amendment"]),
                ("security", ["security", "incident", "token", "credential", "access", "secrets", "secret", "key", "vpn", "badge"]),
                ("staffing", ["staffing", "headcount", "resourcing", "owner", "handoff", "delegate", "approver"]),
                ("scheduling", ["schedule", "scheduled", "calendar", "workshop", "sync", "review", "handoff window"]),
                ("privacy", ["restricted", "authorized", "confidential", "separate", "segregated", "boundary", "need-to-know"]),
                ("deletion", ["delete", "deleted", "remove", "revoke", "revoked", "cache", "history", "purge"]),
            ]
        elif self._domain_label == "medical":
            kw_patterns = [
                ("identifier", ["insurance id", "medicare", "ssn", "identifier"]),
                ("contact", ["address", "phone", "call"]),
                ("lab", ["lab", "result", "mmol/l", "ratio", "reactive"]),
                ("allergy", ["allergy", "hives", "rash"]),
                ("medication", ["medication", "dose", "mg", "tablet", "daily", "weekly"]),
                ("pharmacy", ["pharmacy", "refill"]),
                ("appointment", ["appointment", "follow-up", "follow up", "scheduled", "time"]),
                ("privacy", ["privacy", "restricted", "authorized", "access", "chart separate"]),
                ("deletion", ["delete", "deleted", "remove", "cache", "history"]),
                ("care_plan", ["plan", "monitor", "symptoms", "precautions", "summary"]),
            ]
        else:
            kw_patterns = [
                ("identifier", ["identifier", "id", "reference"]),
                ("contact", ["address", "phone", "call", "email"]),
                ("schedule", ["appointment", "meeting", "scheduled", "time", "calendar"]),
                ("privacy", ["privacy", "restricted", "authorized", "access", "separate"]),
                ("deletion", ["delete", "deleted", "remove", "cache", "history"]),
            ]
        keywords: List[str] = []
        for cat, pats in kw_patterns:
            hit_terms = [p for p in pats if p in lower]
            if hit_terms:
                categories.append(cat)
                keywords.extend(hit_terms)

        # record refs often encode salient labels; include them
        for rr in (turn.record_refs or []):
            rr_lower = str(rr).lower()
            keywords.append(str(rr))
            if "ssn" in rr_lower or "insurance" in rr_lower or "medicare" in rr_lower:
                categories.append("identifier")
            if "lab" in rr_lower or "test" in rr_lower:
                categories.append("lab")
            if "med" in rr_lower:
                categories.append("medication")
            if "followup" in rr_lower:
                categories.append("appointment")
            if any(tok in rr_lower for tok in ["budget", "invoice", "quote", "pricing", "po"]):
                categories.append("finance")
            if any(tok in rr_lower for tok in ["contract", "msa", "sow", "nda", "dpa", "legal"]):
                categories.append("contract")
            if any(tok in rr_lower for tok in ["incident", "token", "credential", "secret", "vpn", "badge"]):
                categories.append("security")
            if any(tok in rr_lower for tok in ["project", "program", "roadmap", "milestone"]):
                categories.append("project")

        # memory op signals
        for op in (turn.memory_ops or []):
            opn = str(op.get("op", ""))
            if opn:
                keywords.append(opn)
                if opn == "delete":
                    categories.append("deletion")
                    rid = op.get("record_id")
                    if rid:
                        keywords.append(str(rid))

        return {
            "summary": summary,
            "keywords": self._dedup_keep_order([k for k in keywords if k]),
            "entities": entities,
            "categories": self._dedup_keep_order(categories),
        }

    def _analyze_turn_metadata_llm(self, turn: Turn) -> Optional[Dict[str, Any]]:
        if self.llm_router is None:
            return None
        prompt = (
            f"You are extracting agentic memory metadata for a single multi-party {self._domain_label} interaction turn.\n"
            "Return STRICT JSON only with keys: summary (string), keywords (array of strings), "
            "entities (array of principal_ids or names), categories (array of short tags).\n"
            "Keep summary <= 30 words. Prefer concrete phrases. Include deletion/privacy tags when applicable.\n\n"
            f"turn_id: {turn.turn_id}\n"
            f"speaker_principal_id: {turn.speaker_principal_id}\n"
            f"speaker_role: {turn.speaker_role}\n"
            f"record_refs: {json.dumps(turn.record_refs or [], ensure_ascii=False)}\n"
            f"memory_ops: {json.dumps(turn.memory_ops or [], ensure_ascii=False)}\n"
            f"text: {turn.text}\n"
        )
        try:
            res = self.llm_router.complete_result(system_prompt="", user_prompt=prompt)
        except Exception as e:
            self.logger.warning("A-Mem metadata LLM call failed for %s: %s", turn.turn_id, e)
            return None
        self._accum_usage(self._meta_llm_usage_accum, getattr(res, "usage", {}) or {})
        parsed = self._parse_json_loose(getattr(res, "text", "") or "")
        if not isinstance(parsed, dict):
            return None
        return {
            "summary": str(parsed.get("summary") or "").strip(),
            "keywords": self._norm_str_list(parsed.get("keywords")),
            "entities": self._norm_str_list(parsed.get("entities")),
            "categories": self._norm_str_list(parsed.get("categories")),
        }

    def _evolve_links(self, node: AMemoryNode) -> None:
        """Link new memory to prior semantically similar memories (A-Mem-style evolution)."""
        if self.link_top_m <= 0 or len(self._nodes) <= 1:
            return
        # Candidate pool excludes self
        candidates = self._nodes[:-1]
        if not candidates:
            return

        query_text = node.retrieval_text()
        if isinstance(self.retriever, EmbeddingRetriever):
            hits, _, _ = self.retriever.search(query_text, top_k=min(self.link_top_m + 2, len(self._nodes)))
        else:
            hits, _ = self.retriever.search(query_text, top_k=min(self.link_top_m + 2, len(self._nodes)))  # type: ignore[misc]

        links: List[str] = []
        for h in hits:
            if h.chunk_id == node.mem_id:
                continue
            if h.score < self.link_score_threshold:
                continue
            links.append(h.chunk_id)
            if len(links) >= self.link_top_m:
                break

        if not links:
            return
        node.linked_ids = self._dedup_keep_order(node.linked_ids + links)
        for other in links:
            self._adj.setdefault(node.mem_id, [])
            self._adj.setdefault(other, [])
            if other not in self._adj[node.mem_id]:
                self._adj[node.mem_id].append(other)
            if node.mem_id not in self._adj[other]:
                self._adj[other].append(node.mem_id)
            other_node = self._nodes_by_id.get(other)
            if other_node and node.mem_id not in other_node.linked_ids:
                other_node.linked_ids.append(node.mem_id)

    # -------------------- query / retrieval --------------------

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        if isinstance(self.retriever, EmbeddingRetriever):
            hits, retrieval_s, q_usage = self.retriever.search(checkpoint.query_text, top_k=self.top_k)
        else:
            hits, retrieval_s = self.retriever.search(checkpoint.query_text, top_k=self.top_k)  # type: ignore[misc]
            q_usage = {}

        # Agentic graph expansion
        expanded = self._expand_via_graph(hits)

        # lightweight rerank using requester role/entity overlap and recency
        reranked = self._rerank_hits(checkpoint, expanded)
        final_hits = reranked[: self.top_k]

        retrieved_memory: List[Dict[str, Any]] = []
        for h in final_hits:
            meta = h.metadata or {}
            node = self._nodes_by_id.get(h.chunk_id)
            text = node.text if node else h.text
            # expose enriched text in side field for debugging, but use raw text as memory text for answering
            retrieved_memory.append(
                {
                    "record_id": h.chunk_id,
                    "turn_id": str(meta.get("turn_id") or (node.turn_id if node else "")),
                    "principal_id": str(meta.get("principal_id") or (node.principal_id if node else "")),
                    "role": str(meta.get("role") or (node.role if node else "")),
                    "text": text,
                    "record_refs": list(meta.get("record_refs") or (node.record_refs if node else [])),
                    "score": float(h.score),
                    "amem_summary": str(meta.get("summary") or (node.summary if node else "")),
                    "amem_keywords": list(meta.get("keywords") or (node.keywords if node else [])),
                    "amem_entities": list(meta.get("entities") or (node.entities if node else [])),
                    "amem_categories": list(meta.get("categories") or (node.categories if node else [])),
                    "amem_links": list(node.linked_ids if node else []),
                }
            )

        native_cards = self._build_native_cards(final_hits)
        rendered_override = None
        if self.answer_protocol == "native":
            rendered_override = self._render_native_memory_block(native_cards)

        out = self._run_llm(
            checkpoint=checkpoint,
            retrieved_memory=retrieved_memory,
            rendered_memory_block_override=rendered_override,
        )
        out["amem_native_cards"] = native_cards
        out["retrieval_s"] = retrieval_s
        out["amem_retrieved_before_rerank"] = len(hits)
        out["amem_retrieved_after_graph_expand"] = len(expanded)
        if q_usage:
            out["embedding_query_usage"] = q_usage
        if self.retrieval_backend == "embedding":
            out["embedding_doc_usage_accum"] = dict(self._embed_usage_accum)
        if self.metadata_mode == "llm":
            out["amem_metadata_llm_usage_accum"] = dict(self._meta_llm_usage_accum)
        return out

    def _build_native_cards(self, final_hits: List[Any]) -> List[Dict[str, Any]]:
        final_ids = {h.chunk_id for h in final_hits}
        cards: List[Dict[str, Any]] = []
        for h in final_hits:
            node = self._nodes_by_id.get(h.chunk_id)
            if node is None:
                continue
            linked_hits = []
            if self.native_include_links and self.native_max_links_per_item > 0:
                for linked_id in node.linked_ids:
                    if linked_id not in final_ids:
                        continue
                    linked_node = self._nodes_by_id.get(linked_id)
                    if linked_node is None:
                        continue
                    linked_hits.append({
                        "mem_id": linked_node.mem_id,
                        "turn_id": linked_node.turn_id,
                        "principal_id": linked_node.principal_id,
                    })
                    if len(linked_hits) >= self.native_max_links_per_item:
                        break
            cards.append(
                {
                    "mem_id": node.mem_id,
                    "turn_id": node.turn_id,
                    "principal_id": node.principal_id,
                    "role": node.role,
                    "timestamp": node.timestamp,
                    "score": float(h.score),
                    "raw_text": node.text,
                    "summary": node.summary,
                    "keywords": list(node.keywords),
                    "entities": list(node.entities),
                    "categories": list(node.categories),
                    "record_refs": list(node.record_refs),
                    "linked_hits": linked_hits,
                }
            )
        return cards

    @staticmethod
    def _render_native_memory_block(cards: List[Dict[str, Any]]) -> str:
        if not cards:
            return "(none)"
        lines: List[str] = [
            "[A-MEM MEMORY FORMAT]",
            "Each memory item contains the original interaction plus structured agentic metadata.",
            "Use the original interaction text as primary evidence; use summary/entities/categories/links only as supporting memory structure.",
            "",
        ]
        for i, card in enumerate(cards, 1):
            lines.append(f"Memory {i}")
            lines.append(f"- speaker: {card.get('principal_id') or 'unknown'}")
            lines.append(f"- role: {card.get('role') or 'unknown'}")
            if card.get("turn_id"):
                lines.append(f"- turn_id: {card['turn_id']}")
            if card.get("timestamp"):
                lines.append(f"- timestamp: {card['timestamp']}")
            lines.append(f"- relevance_score: {float(card.get('score') or 0.0):.4f}")
            lines.append(f"- original_text: {(card.get('raw_text') or '').strip()}")
            summary = str(card.get("summary") or "").strip()
            if summary:
                lines.append(f"- summary: {summary}")
            entities = list(card.get("entities") or [])
            if entities:
                lines.append("- entities: " + ", ".join(map(str, entities)))
            categories = list(card.get("categories") or [])
            if categories:
                lines.append("- categories: " + ", ".join(map(str, categories)))
            keywords = list(card.get("keywords") or [])
            if keywords:
                lines.append("- keywords: " + ", ".join(map(str, keywords)))
            record_refs = list(card.get("record_refs") or [])
            if record_refs:
                lines.append("- record_refs: " + ", ".join(map(str, record_refs)))
            linked_hits = list(card.get("linked_hits") or [])
            if linked_hits:
                rel_ids = [str(it.get("turn_id") or it.get("mem_id") or "") for it in linked_hits]
                rel_ids = [x for x in rel_ids if x]
                if rel_ids:
                    lines.append("- related_memories_in_context: " + ", ".join(rel_ids))
            lines.append("")
        return "\n".join(lines).strip()

    def _expand_via_graph(self, hits: List[Any]) -> List[Any]:
        if self.graph_expand_hops <= 0 or self.graph_expand_per_hit <= 0:
            return list(hits)
        # Reuse Retrieved dataclass shape from retrievers (duck-typing)
        out = list(hits)
        seen = {h.chunk_id for h in hits}
        frontier = list(hits)
        for hop in range(self.graph_expand_hops):
            new_frontier: List[Any] = []
            for h in frontier:
                nbrs = self._adj.get(h.chunk_id, [])[: self.graph_expand_per_hit]
                for nid in nbrs:
                    if nid in seen:
                        continue
                    node = self._nodes_by_id.get(nid)
                    if not node:
                        continue
                    # Use decayed inherited score for graph expansion
                    inherited_score = float(h.score) * (0.85 ** (hop + 1))
                    class _Hit:  # local lightweight object
                        def __init__(self, chunk_id, score, text, metadata):
                            self.chunk_id = chunk_id
                            self.score = score
                            self.text = text
                            self.metadata = metadata
                    nh = _Hit(
                        chunk_id=nid,
                        score=inherited_score,
                        text=node.retrieval_text(),
                        metadata={
                            "turn_id": node.turn_id,
                            "principal_id": node.principal_id,
                            "role": node.role,
                            "record_refs": node.record_refs,
                            "summary": node.summary,
                            "keywords": node.keywords,
                            "entities": node.entities,
                            "categories": node.categories,
                        },
                    )
                    out.append(nh)
                    new_frontier.append(nh)
                    seen.add(nid)
            frontier = new_frontier
            if not frontier:
                break
        return out

    def _rerank_hits(self, checkpoint: Checkpoint, hits: List[Any]) -> List[Any]:
        q = checkpoint.query_text.lower()
        requester_role = (checkpoint.asker_role or "").lower()
        # Principal/alias tokens from query for overlap bonus
        q_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z\-_/]{2,}", q))

        scored: List[Tuple[float, Any]] = []
        for h in hits:
            meta = h.metadata or {}
            s = float(h.score)
            role = str(meta.get("role") or "").lower()
            if requester_role and role == requester_role:
                s += self.rerank_role_bonus
            entities = [str(x).lower() for x in (meta.get("entities") or [])]
            if entities:
                overlap = 0
                for e in entities:
                    e_toks = set(re.findall(r"[a-zA-Z][a-zA-Z\-_/]{2,}", e))
                    if q_tokens & e_toks:
                        overlap = 1
                        break
                if overlap:
                    s += self.rerank_entity_bonus
            # recency bonus from tNNN (encourage recent memories slightly)
            tid = str(meta.get("turn_id") or "")
            tnum = self._turn_num(tid)
            if tnum > 0:
                s += min(0.05, 0.0005 * tnum)
            scored.append((s, h))

        scored.sort(key=lambda x: x[0], reverse=True)
        # write back adjusted score for transparency
        out: List[Any] = []
        seen = set()
        for s, h in scored:
            if h.chunk_id in seen:
                continue
            try:
                h.score = float(s)
            except Exception:
                pass
            out.append(h)
            seen.add(h.chunk_id)
        return out

    # -------------------- helpers --------------------

    @staticmethod
    def _turn_num(turn_id: str) -> int:
        m = re.search(r"t(\d+)", str(turn_id))
        if not m:
            return -1
        try:
            return int(m.group(1))
        except Exception:
            return -1

    @staticmethod
    def _dedup_keep_order(items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in items:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    @staticmethod
    def _norm_str_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            items = v
        else:
            items = [v]
        out: List[str] = []
        for x in items:
            s = str(x).strip()
            if s:
                out.append(s)
        return AMemAgent._dedup_keep_order(out)

    @staticmethod
    def _accum_usage(acc: Dict[str, float], usage: Dict[str, Any]) -> None:
        for k in ["input_tokens", "output_tokens", "total_tokens"]:
            v = usage.get(k)
            if isinstance(v, (int, float)):
                acc[k] = float(acc.get(k, 0.0)) + float(v)

    @staticmethod
    def _parse_json_loose(text: str) -> Any:
        s = (text or "").strip()
        if not s:
            return None
        # strip markdown fences
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```$", "", s)
        # direct json
        try:
            return json.loads(s)
        except Exception:
            pass
        # extract first object/array span
        start_obj = s.find("{")
        start_arr = s.find("[")
        starts = [i for i in [start_obj, start_arr] if i >= 0]
        if starts:
            start = min(starts)
            # try progressively shorter substrings ending at matching braces is complex; use last brace/bracket heuristic
            candidates = []
            end_obj = s.rfind("}")
            end_arr = s.rfind("]")
            if end_obj > start:
                candidates.append(s[start:end_obj+1])
            if end_arr > start:
                candidates.append(s[start:end_arr+1])
            for c in candidates:
                try:
                    return json.loads(c)
                except Exception:
                    try:
                        return ast.literal_eval(c)
                    except Exception:
                        pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return None
