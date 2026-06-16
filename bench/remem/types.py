from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryNode:
    node_id: str
    node_type: str  # verbatim|gists|facts|entity
    content: str
    source_turn_id: str
    principal_id: str
    role: str
    timestamp: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    node_id: str
    node_type: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolTrace:
    step: int
    tool: str
    reasoning: str
    parameters: Dict[str, Any]
    observation: str
    results: List[SearchResult] = field(default_factory=list)


@dataclass
class QueryState:
    query: str
    max_steps: int
    step: int = 0
    visited_nodes: set[str] = field(default_factory=set)
    traces: List[ToolTrace] = field(default_factory=list)
    last_focus_nodes: List[str] = field(default_factory=list)
    final_answer_requested: bool = False


@dataclass
class EvidencePacket:
    verbatim: SearchResult
    supporting_gists: List[SearchResult] = field(default_factory=list)
    supporting_facts: List[SearchResult] = field(default_factory=list)
    supporting_entities: List[SearchResult] = field(default_factory=list)
