# Adding a New Memory Agent

GateMem provides a small agent interface so that new memory methods can be evaluated under the same incremental protocol as the built-in baselines.

## 1. Implement the agent

Create a new file under `bench/agents/`, for example `bench/agents/my_agent.py`.

A GateMem agent should inherit from `BaseMemoryAgent` and implement three methods:

```python
from __future__ import annotations

from typing import Any, Dict, List

from bench.agents.base import BaseMemoryAgent, Checkpoint, Turn


class MyAgent(BaseMemoryAgent):
    def reset(self, episode: Dict[str, Any]) -> None:
        super().reset(episode)
        self.turns: List[Turn] = []

    def ingest(self, turn: Turn) -> None:
        self.turns.append(turn)

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        return {
            "action": "answer",              # answer | answer_redacted | refuse | no_memory
            "answer": "...",                # natural-language response
            "answer_structured": {},          # optional structured response
            "used_record_ids": [],            # optional audit/debug ids
        }
```

The benchmark calls these methods as follows:

1. `reset(episode)` at the beginning of each episode.
2. `ingest(turn)` for each turn up to a checkpoint boundary.
3. `query(checkpoint)` when the agent is evaluated at a hidden checkpoint.

The agent does **not** receive hidden labels such as `query_type`, `expected_action`, `judge_spec`, or `leak_targets` as part of the user-facing task. Those fields are only used by the evaluator.

## 2. Register the agent

Add the class to `bench/agents/__init__.py`:

```python
from .my_agent import MyAgent

AGENT_REGISTRY = {
    ...,
    "my_agent": MyAgent,
}
```

After registration, the agent can be selected with `--agent my_agent`.

## 3. Run a single evaluation

```bash
python bench/scripts/run_eval.py \
  --config configs/runs/paper_main.yaml \
  --data_dir bench/data/medical \
  --agent my_agent \
  --llm_provider openai \
  --llm_model gpt-4o-mini \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o \
  --run_name my_agent_medical_gpt4omini
```

## 4. Add the agent to a sweep

You can add a new baseline entry to a sweep config such as `configs/sweeps/paper_matrix.yaml`:

```yaml
baselines:
  my_agent:
    agent: my_agent
```

Then run:

```bash
python scripts/sweep.py \
  --config configs/sweeps/paper_matrix.yaml \
  --domains medical \
  --models gpt4omini \
  --baselines my_agent
```

## 5. Minimal example

The repository includes a minimal example implementation:

```text
bench/agents/example_agent.py
```

It is intentionally conservative and always refuses. It is not a competitive baseline; it only demonstrates the interface. You can run it with:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent example \
  --run_name example_agent_demo
```
