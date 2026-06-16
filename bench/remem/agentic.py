from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .prompts import TOOL_SELECTION_SYSTEM_PROMPT, render_tool_selection_user_prompt
from .types import QueryState, SearchResult


class ToolSelector:
    def __init__(self, *, llm_router: Optional[Any], logger: Any = None) -> None:
        self.llm_router = llm_router
        self.logger = logger

    @property
    def use_llm(self) -> bool:
        return self.llm_router is not None and getattr(self.llm_router, "provider", "stub") != "stub"

    def select(
        self,
        *,
        state: QueryState,
        recent_gists: List[Dict[str, Any]],
        recent_entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.use_llm:
            user_prompt = render_tool_selection_user_prompt(
                query=state.query,
                previous_steps=[
                    {
                        "step": tr.step,
                        "tool": tr.tool,
                        "reasoning": tr.reasoning,
                        "parameters": tr.parameters,
                        "observation": tr.observation,
                        "results": [
                            {
                                "node_id": r.node_id,
                                "node_type": r.node_type,
                                "content": r.content[:220],
                                "score": round(float(r.score), 4),
                            }
                            for r in tr.results[:10]
                        ],
                    }
                    for tr in state.traces
                ],
                recent_gists=recent_gists,
                recent_entities=recent_entities,
                step=state.step + 1,
                max_steps=state.max_steps,
            )
            res = self.llm_router.complete_result(
                system_prompt=TOOL_SELECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            obj = self._parse_json_dict(res.text)
            fn = str(obj.get("function") or "").strip()
            if fn in {
                "semantic_retrieve",
                "lexical_retrieve",
                "find_gist_contexts",
                "find_entity_contexts",
                "output_answer",
            }:
                params = obj.get("parameters") if isinstance(obj.get("parameters"), dict) else {}
                if state.step == 0 and fn in {"semantic_retrieve", "lexical_retrieve"}:
                    params["query"] = state.query
                reasoning = str(obj.get("reasoning") or "")
                return {"tool": fn, "parameters": params, "reasoning": reasoning}
            if self.logger:
                self.logger.warning("ReMem tool-selection parse failed; falling back to heuristic")
        return self._heuristic_select(state=state, recent_gists=recent_gists, recent_entities=recent_entities)

    def _heuristic_select(
        self,
        *,
        state: QueryState,
        recent_gists: List[Dict[str, Any]],
        recent_entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        q = state.query.lower()
        if not state.traces:
            has_rare = bool(re.search(r"\b[A-Z]{2,}\d+|\d{3,}|[A-Za-z]+_[A-Za-z0-9_]+\b", state.query))
            tool = "lexical_retrieve" if has_rare else "semantic_retrieve"
            return {"tool": tool, "parameters": {"query": state.query}, "reasoning": "Initial retrieval."}
        if state.step >= state.max_steps - 1:
            return {"tool": "output_answer", "parameters": {}, "reasoning": "Maximum steps reached."}
        if state.traces:
            last_tool = state.traces[-1].tool
            if last_tool in {"find_gist_contexts", "find_entity_contexts"}:
                return {"tool": "output_answer", "parameters": {}, "reasoning": "One focused expansion has already been performed."}
            if any(r.node_type == "verbatim" for r in state.traces[-1].results):
                return {"tool": "output_answer", "parameters": {}, "reasoning": "Direct supporting evidence is already present."}
        if recent_gists:
            return {
                "tool": "find_gist_contexts",
                "parameters": {"gist_id": 1},
                "reasoning": "Expand the strongest retrieved gist.",
            }
        if recent_entities:
            return {
                "tool": "find_entity_contexts",
                "parameters": {"entity_id": 1},
                "reasoning": "Expand the strongest retrieved entity.",
            }
        if any(tok in q for tok in ["when", "before", "after", "current", "latest", "now"]):
            return {
                "tool": "semantic_retrieve",
                "parameters": {"query": state.query},
                "reasoning": "Need semantic evidence for temporal/state reasoning.",
            }
        return {"tool": "output_answer", "parameters": {}, "reasoning": "Available evidence is likely sufficient."}

    @staticmethod
    def _parse_json_dict(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", text)
            text = re.sub(r"\n```\s*$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        js = text[start : end + 1]
        try:
            obj = json.loads(js)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            js = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', js)
            js = js.replace("'", '"')
            try:
                obj = json.loads(js)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}


def summarize_recent_candidates(results: List[SearchResult], *, prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    idx = 0
    for row in results:
        if row.node_type != prefix:
            continue
        idx += 1
        out.append({"id": idx, "node_id": row.node_id, "content": row.content[:220]})
    return out


def make_observation(results: List[SearchResult], *, tool: str) -> str:
    if not results:
        return f"{tool}: no relevant memory found"
    type_counts: Dict[str, int] = {}
    for r in results:
        type_counts[r.node_type] = type_counts.get(r.node_type, 0) + 1
    parts = [f"{v} {k}" for k, v in sorted(type_counts.items())]
    return f"{tool}: retrieved " + ", ".join(parts)
