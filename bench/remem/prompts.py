from __future__ import annotations

import json
from typing import Any, Dict, List


GIST_SYSTEM_PROMPT = """You are a meticulous information extractor.
Convert one conversational turn into a list of concise episodic gists.
Rules:
- Return valid JSON only: {"gists": [...]}.
- Split compound statements into atomic gists.
- Preserve concrete participants, objects, locations, quantities, and states.
- Prefix each gist with the turn timestamp in square brackets when available.
- Resolve obvious relative time expressions using the provided timestamp when possible, but do not invent details.
- Keep each gist self-contained and retrieval-friendly.
"""


FACT_SYSTEM_PROMPT = """Extract structured facts from one conversational turn.
Return valid JSON only with shape: {"facts": [{"subject": str, "predicate": str, "object": str, "qualifiers": {...}}]}.
Rules:
- Include `record_time` in qualifiers when a timestamp is provided.
- Prefer reusable event handles and concise entities.
- Use only facts supported by the text.
- If no clean fact can be formed, return an empty list.
"""


TOOL_SELECTION_SYSTEM_PROMPT = """You are controlling a ReMem-style episodic memory agent.
Pick the single best next tool and return valid JSON only:
{"reasoning": str, "function": str, "parameters": {...}}

Allowed tools:
- semantic_retrieve(query): retrieve semantically relevant gists, facts, and entities.
- lexical_retrieve(query): retrieve exact / keyword-overlap matches.
- find_gist_contexts(gist_id): expand one previously retrieved gist into related gists, facts, and verbatim evidence.
- find_entity_contexts(entity_id): expand one previously retrieved entity into connected facts, gists, and verbatim evidence.
- output_answer(): stop searching when enough evidence is gathered.

Guidelines:
1. Usually start with semantic_retrieve unless the question depends on exact identifiers / exact strings.
2. Prefer find_gist_contexts over find_entity_contexts when both are plausible.
3. Use output_answer only when the retrieved evidence already supports the answer.
4. Keep parameters minimal and valid.
5. Never explain outside JSON.

Example:
{"reasoning": "The question asks for the current project state, so I should first gather the most relevant episodic memories.", "function": "semantic_retrieve", "parameters": {"query": "What is the current project state and blocker?"}}
"""


def render_gist_user_prompt(*, timestamp: str | None, principal_id: str, role: str, text: str) -> str:
    ts = timestamp or "unknown"
    return (
        f"Timestamp: {ts}\n"
        f"Speaker principal_id: {principal_id}\n"
        f"Speaker role: {role}\n"
        f"Turn text:\n{text}\n"
    )


def render_fact_user_prompt(
    *, timestamp: str | None, principal_id: str, role: str, text: str, gists: List[str]
) -> str:
    ts = timestamp or "unknown"
    gist_block = "\n".join(f"- {g}" for g in gists) if gists else "(none)"
    return (
        f"Timestamp: {ts}\n"
        f"Speaker principal_id: {principal_id}\n"
        f"Speaker role: {role}\n"
        f"Turn text:\n{text}\n\n"
        f"Previously extracted gists:\n{gist_block}\n"
    )


def render_tool_selection_user_prompt(
    *,
    query: str,
    previous_steps: List[Dict[str, Any]],
    recent_gists: List[Dict[str, Any]],
    recent_entities: List[Dict[str, Any]],
    step: int,
    max_steps: int,
) -> str:
    prev_block = json.dumps(previous_steps[-3:], ensure_ascii=False, indent=2) if previous_steps else "[]"
    gist_block = json.dumps(recent_gists[:5], ensure_ascii=False, indent=2) if recent_gists else "[]"
    ent_block = json.dumps(recent_entities[:5], ensure_ascii=False, indent=2) if recent_entities else "[]"
    return (
        f"Question: {query}\n"
        f"Current step: {step} / {max_steps}\n\n"
        f"Previous steps and retrieved memory summaries:\n{prev_block}\n\n"
        f"Recent gist candidates (use their numeric id for gist_id):\n{gist_block}\n\n"
        f"Recent entity candidates (use their numeric id for entity_id):\n{ent_block}\n"
    )
