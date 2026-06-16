from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseMemoryAgent, Turn, Checkpoint


class LongContextAgent(BaseMemoryAgent):
    """Baseline: concatenate all ingested turns as context (bounded by max_turns)."""

    def __init__(
        self,
        *,
        max_turns: int = 50,
        llm_mode: str = "leaky",
        llm_router: Any = None,
        query_prompt_path: str | None = None,
    ) -> None:
        super().__init__(top_k=max_turns, llm_mode=llm_mode, llm_router=llm_router, query_prompt_path=query_prompt_path)
        self.max_turns = max_turns
        self._turns: List[Turn] = []

    def reset(self, episode: Dict[str, Any]) -> None:
        super().reset(episode)
        self._turns = []

    def ingest(self, turn: Turn) -> None:
        self._turns.append(turn)

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        ctx_turns = self._turns[-self.max_turns :]

        retrieved_memory: List[Dict[str, Any]] = []
        for t in ctx_turns:
            record_id = (t.record_refs[0] if t.record_refs else t.turn_id)
            retrieved_memory.append(
                {
                    "record_id": record_id,
                    "turn_id": t.turn_id,
                    "principal_id": t.speaker_principal_id,
                    "role": t.speaker_role,
                    "text": t.text,
                    "record_refs": t.record_refs or [],
                }
            )

        out = self._run_llm(checkpoint=checkpoint, retrieved_memory=retrieved_memory)
        out["retrieval_s"] = 0.0
        out["context_turns"] = len(ctx_turns)
        return out
