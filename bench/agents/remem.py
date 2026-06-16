from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .base import BaseMemoryAgent, Checkpoint, Turn
from bench.domains import detect_domain_from_episode

from bench.remem.agentic import ToolSelector, make_observation, summarize_recent_candidates
from bench.remem.extractor import EpisodicExtractor
from bench.remem.retriever import ReMemIndex
from bench.remem.types import EvidencePacket, QueryState, SearchResult, ToolTrace


class ReMemAgent(BaseMemoryAgent):
    """REMem-style episodic memory baseline."""

    def __init__(
        self,
        *,
        top_k: int = 10,
        llm_mode: str = "leaky",
        llm_router: Any = None,
        query_prompt_path: str | None = None,
        embed_router: Optional[Any] = None,
        variant: str = "iterative",
        max_steps: int = 5,
        retrieval_top_k: Optional[int] = 10,
        linking_top_k: Optional[int] = 5,
        qa_top_k: Optional[int] = None,
        synonymy_threshold: float = 0.8,
        logger: Any = None,
        answer_protocol: str = "standard",
        native_max_gists_per_packet: int = 2,
        native_max_facts_per_packet: int = 2,
        native_max_entities_per_packet: int = 2,
    ) -> None:
        super().__init__(
            top_k=top_k,
            llm_mode=llm_mode,
            llm_router=llm_router,
            query_prompt_path=query_prompt_path,
            logger=logger,
            answer_protocol=answer_protocol,
        )
        variant = (variant or "iterative").lower().strip()
        if variant not in {"iterative", "single"}:
            raise ValueError("ReMem variant must be iterative|single")

        self.variant = variant
        self.max_steps = max(1, int(max_steps))
        self.retrieval_top_k = max(1, int(retrieval_top_k if retrieval_top_k is not None else max(top_k, 10)))
        self.linking_top_k = max(1, int(linking_top_k if linking_top_k is not None else min(max(top_k, 5), self.retrieval_top_k)))
        default_qa_top_k = top_k if qa_top_k is None else int(qa_top_k)
        self.qa_top_k = max(1, min(int(default_qa_top_k), int(top_k)))

        self.index = ReMemIndex(embed_router=embed_router, logger=self.logger, synonymy_threshold=float(synonymy_threshold))
        self.extractor = EpisodicExtractor(llm_router=llm_router, logger=self.logger)
        self.selector = ToolSelector(llm_router=llm_router, logger=self.logger)
        self.native_max_gists_per_packet = max(0, int(native_max_gists_per_packet))
        self.native_max_facts_per_packet = max(0, int(native_max_facts_per_packet))
        self.native_max_entities_per_packet = max(0, int(native_max_entities_per_packet))

        self._domain_key = "generic"
        self._turn_counter = 0

    def reset(self, episode: Dict[str, Any]) -> None:
        super().reset(episode)
        self.index.clear()
        self._domain_key = detect_domain_from_episode(episode)
        self._turn_counter = 0

    def ingest(self, turn: Turn) -> None:
        self._turn_counter += 1
        merged_refs = self._merge_record_refs(turn.record_refs, turn.text)
        verbatim_text = self._format_verbatim(turn)
        gists, facts, entities = self.extractor.extract(timestamp=turn.timestamp, principal_id=turn.speaker_principal_id, role=turn.speaker_role, text=turn.text)
        self.index.add_turn(
            turn_id=turn.turn_id,
            principal_id=turn.speaker_principal_id,
            role=turn.speaker_role,
            timestamp=turn.timestamp,
            verbatim_text=verbatim_text,
            gists=gists,
            facts=facts,
            entities=entities,
            record_refs=merged_refs,
        )

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        t0 = time.perf_counter()
        state = QueryState(query=checkpoint.query_text, max_steps=self.max_steps)
        scored_nodes: Dict[str, float] = {}
        usage_acc = {"input_tokens": 0.0, "total_tokens": 0.0}

        if self.variant == "single":
            results, usage = self.index.semantic_retrieve(checkpoint.query_text, top_k=self.retrieval_top_k)
            for k, v in usage.items():
                if isinstance(v, (int, float)):
                    usage_acc[k] += float(v)
            self._merge_results_into_state(state=state, scored_nodes=scored_nodes, tool="semantic_retrieve", reasoning="Single-step retrieval.", parameters={"query": checkpoint.query_text}, results=results)
        else:
            for step in range(self.max_steps):
                state.step = step
                recent_gists = self._collect_recent_candidates(state.traces, node_type="gists")
                recent_entities = self._collect_recent_candidates(state.traces, node_type="entity")
                choice = self.selector.select(state=state, recent_gists=recent_gists, recent_entities=recent_entities)
                tool = str(choice.get("tool") or "output_answer")
                params = choice.get("parameters") if isinstance(choice.get("parameters"), dict) else {}
                reasoning = str(choice.get("reasoning") or "")

                if tool == "output_answer":
                    state.final_answer_requested = True
                    state.traces.append(ToolTrace(step=step + 1, tool=tool, reasoning=reasoning, parameters=params, observation="output_answer: stop exploration", results=[]))
                    break

                results: List[SearchResult] = []
                if tool == "semantic_retrieve":
                    query = str(params.get("query") or checkpoint.query_text)
                    results, usage = self.index.semantic_retrieve(query, top_k=self.retrieval_top_k, exclude_ids=state.visited_nodes)
                    for k, v in usage.items():
                        if isinstance(v, (int, float)):
                            usage_acc[k] += float(v)
                elif tool == "lexical_retrieve":
                    query = str(params.get("query") or checkpoint.query_text)
                    results = self.index.lexical_retrieve(query, top_k=self.retrieval_top_k, exclude_ids=state.visited_nodes)
                elif tool == "find_gist_contexts":
                    gist_ref = params.get("gist_id")
                    gist_node_id = self._resolve_focus_ref(state.traces, node_type="gists", ref=gist_ref)
                    if gist_node_id:
                        results = self.index.expand_gist(gist_node_id, query=checkpoint.query_text, limit=self.linking_top_k)
                elif tool == "find_entity_contexts":
                    entity_ref = params.get("entity_id")
                    entity_node_id = self._resolve_focus_ref(state.traces, node_type="entity", ref=entity_ref)
                    if entity_node_id:
                        results = self.index.expand_entity(entity_node_id, query=checkpoint.query_text, limit=self.linking_top_k)
                else:
                    state.traces.append(ToolTrace(step=step + 1, tool=tool, reasoning=reasoning, parameters=params, observation=f"unknown tool: {tool}", results=[]))
                    break

                self._merge_results_into_state(state=state, scored_nodes=scored_nodes, tool=tool, reasoning=reasoning, parameters=params, results=results)
                if tool in {"find_gist_contexts", "find_entity_contexts"} and not results:
                    break

        evidence = self.index.collect_evidence(scored_nodes, top_k=self.qa_top_k)
        retrieved_memory = [self._search_result_to_memory_dict(r) for r in evidence]
        packets = self.index.collect_evidence_packets(
            scored_nodes,
            top_k=self.qa_top_k,
            max_gists_per_packet=self.native_max_gists_per_packet,
            max_facts_per_packet=self.native_max_facts_per_packet,
            max_entities_per_packet=self.native_max_entities_per_packet,
        )
        rendered_override = None
        if self.answer_protocol == "native":
            rendered_override = self._render_native_memory_block(packets)
        out = self._run_llm(
            checkpoint=checkpoint,
            retrieved_memory=retrieved_memory,
            rendered_memory_block_override=rendered_override,
        )
        out["retrieval_s"] = time.perf_counter() - t0
        out["remem_variant"] = self.variant
        out["remem_effective_top_k"] = self.qa_top_k
        out["remem_retrieval_top_k"] = self.retrieval_top_k
        out["remem_linking_top_k"] = self.linking_top_k
        out["remem_evidence_packets"] = [self._packet_to_dict(p) for p in packets]
        out["remem_trace"] = [
            {
                "step": tr.step,
                "tool": tr.tool,
                "reasoning": tr.reasoning,
                "parameters": tr.parameters,
                "observation": tr.observation,
                "results": [
                    {"node_id": r.node_id, "node_type": r.node_type, "score": r.score, "content": r.content}
                    for r in tr.results
                ],
            }
            for tr in state.traces
        ]
        out["remem_embed_usage"] = usage_acc
        out["remem_nodes_indexed"] = len(self.index.nodes)
        return out

    @staticmethod
    def _packet_to_dict(packet: EvidencePacket) -> Dict[str, Any]:
        return {
            "verbatim": ReMemAgent._search_result_to_memory_dict(packet.verbatim),
            "supporting_gists": [ReMemAgent._search_result_to_memory_dict(r) for r in packet.supporting_gists],
            "supporting_facts": [ReMemAgent._search_result_to_memory_dict(r) for r in packet.supporting_facts],
            "supporting_entities": [ReMemAgent._search_result_to_memory_dict(r) for r in packet.supporting_entities],
        }

    @staticmethod
    def _render_native_memory_block(packets: List[EvidencePacket]) -> str:
        if not packets:
            return "(none)"
        lines: List[str] = [
            "[REMEM MEMORY FORMAT]",
            "Each memory item includes a verbatim episodic memory plus its linked gist/fact/entity supports.",
            "Use verbatim as the primary evidence. Use gist/fact/entity supports to interpret or disambiguate the episode.",
            "",
        ]
        for i, packet in enumerate(packets, 1):
            vb = packet.verbatim
            meta = vb.metadata or {}
            lines.append(f"Episode Memory {i}")
            lines.append(f"- speaker: {meta.get('principal_id') or 'unknown'}")
            lines.append(f"- role: {meta.get('role') or 'unknown'}")
            if meta.get("turn_id"):
                lines.append(f"- turn_id: {meta.get('turn_id')}")
            if meta.get("timestamp"):
                lines.append(f"- timestamp: {meta.get('timestamp')}")
            lines.append(f"- primary_verbatim: {vb.content}")
            if packet.supporting_gists:
                lines.append("- supporting_gists:")
                for row in packet.supporting_gists:
                    lines.append(f"  - {row.content}")
            if packet.supporting_facts:
                lines.append("- supporting_facts:")
                for row in packet.supporting_facts:
                    lines.append(f"  - {row.content}")
            if packet.supporting_entities:
                lines.append("- supporting_entities:")
                for row in packet.supporting_entities:
                    lines.append(f"  - {row.content}")
            lines.append("")
        return "\n".join(lines).strip()

    def _merge_results_into_state(self, *, state: QueryState, scored_nodes: Dict[str, float], tool: str, reasoning: str, parameters: Dict[str, Any], results: List[SearchResult]) -> None:
        obs = make_observation(results, tool=tool)
        trace = ToolTrace(step=state.step + 1, tool=tool, reasoning=reasoning, parameters=parameters, observation=obs, results=results)
        state.traces.append(trace)
        focus: List[str] = []
        for r in results:
            state.visited_nodes.add(r.node_id)
            prev = scored_nodes.get(r.node_id, 0.0)
            scored_nodes[r.node_id] = max(prev, float(r.score))
            focus.append(r.node_id)
        state.last_focus_nodes = focus

    @staticmethod
    def _collect_recent_candidates(traces: List[ToolTrace], *, node_type: str) -> List[Dict[str, Any]]:
        for tr in reversed(traces):
            rows = summarize_recent_candidates(tr.results, prefix=node_type)
            if rows:
                return rows
        return []

    @staticmethod
    def _resolve_focus_ref(traces: List[ToolTrace], *, node_type: str, ref: Any) -> Optional[str]:
        candidate_steps: List[List[SearchResult]] = [tr.results for tr in reversed(traces) if tr.results]
        direct = str(ref).strip() if ref is not None else ""
        if direct and not direct.isdigit():
            for results in candidate_steps:
                for r in results:
                    if r.node_type == node_type and r.node_id == direct:
                        return r.node_id
        try:
            target_index = max(1, int(ref)) if ref is not None else 1
        except Exception:
            target_index = 1
        for results in candidate_steps:
            typed = [r for r in results if r.node_type == node_type]
            if len(typed) >= target_index:
                return typed[target_index - 1].node_id
        return None

    @staticmethod
    def _format_verbatim(turn: Turn) -> str:
        ts = f"[{turn.timestamp}] " if turn.timestamp else ""
        return f"{ts}[{turn.speaker_role}:{turn.speaker_principal_id}] {turn.text}".strip()

    @staticmethod
    def _search_result_to_memory_dict(row: SearchResult) -> Dict[str, Any]:
        meta = row.metadata or {}
        record_refs = list(meta.get("record_refs") or [])
        return {
            "record_id": record_refs[0] if record_refs else "",
            "turn_id": str(meta.get("turn_id") or ""),
            "principal_id": str(meta.get("principal_id") or ""),
            "text": row.content,
            "node_id": row.node_id,
            "node_type": row.node_type,
            "score": row.score,
            "role": str(meta.get("role") or ""),
            "timestamp": meta.get("timestamp"),
            "record_refs": record_refs,
        }
