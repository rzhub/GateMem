from __future__ import annotations

from typing import Any, Dict, List

from bench.agents.base import BaseMemoryAgent, Checkpoint, Turn


class ExampleAgent(BaseMemoryAgent):
    """Minimal example agent for demonstrating the GateMem agent interface.

    This agent intentionally implements a simple conservative policy: it stores
    ingested turns but always refuses at query time. It is not intended to be a
    competitive baseline. Its purpose is to show the minimum methods a new
    memory method needs to implement in order to run through the benchmark.
    """

    def reset(self, episode: Dict[str, Any]) -> None:
        super().reset(episode)
        self.turns: List[Turn] = []

    def ingest(self, turn: Turn) -> None:
        self.turns.append(turn)

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        return {
            "action": "refuse",
            "answer": "I cannot provide that information.",
            "answer_structured": {},
            "used_record_ids": [],
            "debug": {
                "agent": "example",
                "n_ingested_turns": len(self.turns),
            },
        }
