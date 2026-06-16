from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: str
    text: str
    turn_ids: List[str]
    principal_ids: List[str]
    roles: List[str]
    record_refs: List[str]


class Chunker:
    """Turn -> chunk strategy.

    Modes:
      - turn: each turn is a chunk
      - window: group fixed number of turns
      - chars: group until approx max_chars

    This is intentionally simple (V0). You can swap in token-based chunking later.
    """

    def __init__(
        self,
        mode: str = "turn",
        window_turns: int = 5,
        max_chars: int = 4000,
        *,
        include_speaker_prefix: bool = True,
    ) -> None:
        self.mode = mode
        self.window_turns = max(1, int(window_turns))
        self.max_chars = max(200, int(max_chars))
        self.include_speaker_prefix = include_speaker_prefix

        self._buf_turns: List[Dict[str, Any]] = []
        self._chunk_counter = 0

    def _turn_text(self, turn: Dict[str, Any]) -> str:
        txt = turn.get("text", "")
        if not self.include_speaker_prefix:
            return txt
        sp = turn.get("speaker", {})
        pid = sp.get("principal_id", "unknown")
        role = sp.get("role", "")
        return f"[{role}:{pid}] {txt}".strip()

    def add_turn(self, turn: Dict[str, Any]) -> List[Chunk]:
        """Add a turn and return any flushed chunks."""
        if self.mode == "turn":
            self._chunk_counter += 1
            return [self._make_chunk([turn])]

        self._buf_turns.append(turn)

        flushed: List[Chunk] = []
        if self.mode == "window":
            while len(self._buf_turns) >= self.window_turns:
                group = self._buf_turns[: self.window_turns]
                self._buf_turns = self._buf_turns[self.window_turns :]
                self._chunk_counter += 1
                flushed.append(self._make_chunk(group))
            return flushed

        if self.mode == "chars":
            # flush when buffer text exceeds max_chars
            total = 0
            for t in self._buf_turns:
                total += len(t.get("text", ""))
            if total >= self.max_chars:
                group = self._buf_turns
                self._buf_turns = []
                self._chunk_counter += 1
                flushed.append(self._make_chunk(group))
            return flushed

        raise ValueError(f"Unknown chunk mode: {self.mode}")

    def finalize(self) -> List[Chunk]:
        """Flush remaining turns as one final chunk."""
        if not self._buf_turns:
            return []
        self._chunk_counter += 1
        group = self._buf_turns
        self._buf_turns = []
        return [self._make_chunk(group)]

    def _make_chunk(self, turns: List[Dict[str, Any]]) -> Chunk:
        turn_ids = [t.get("turn_id") for t in turns]
        principal_ids = [t.get("speaker", {}).get("principal_id", "") for t in turns]
        roles = [t.get("speaker", {}).get("role", "") for t in turns]
        record_refs: List[str] = []
        for t in turns:
            refs = t.get("record_refs") or []
            for r in refs:
                if r not in record_refs:
                    record_refs.append(r)

        text = "\n".join(self._turn_text(t) for t in turns)
        cid = f"chunk_{self._chunk_counter:04d}_{turn_ids[0]}_{turn_ids[-1]}"
        return Chunk(
            chunk_id=cid,
            text=text,
            turn_ids=turn_ids,
            principal_ids=principal_ids,
            roles=roles,
            record_refs=record_refs,
        )
