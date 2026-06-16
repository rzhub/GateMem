# GateMem Benchmark Details

This directory contains the core benchmark data, agents, prompts, evaluation scripts, and scoring utilities for GateMem.

GateMem evaluates **multi-principal shared-memory agents** in long-form multi-party interaction episodes. The benchmark asks whether an agent can remain useful while enforcing access boundaries and honoring deletion requests.

For the paper overview, figures, and main results, see the top-level [`README.md`](../README.md).

---

## Contents

- [Overview](#overview)
- [Important Terminology Note](#important-terminology-note)
- [Directory Structure](#directory-structure)
- [Data Format](#data-format)
  - [`episodes.jsonl`](#episodesjsonl)
  - [`checkpoints.jsonl`](#checkpointsjsonl)
- [Evaluation Protocol](#evaluation-protocol)
- [Quickstart](#quickstart)
- [Running with Hosted LLM APIs](#running-with-hosted-llm-apis)
- [Baselines](#baselines)
  - [Baseline Summary](#baseline-summary)
  - [Long-Context](#long-context)
  - [Naive RAG](#naive-rag)
  - [Policy RAG](#policy-rag)
  - [A-Mem](#a-mem)
  - [Mem0](#mem0)
  - [ReMeM](#remem)
  - [Example Agent](#example-agent)
- [Evaluating External Predictions](#evaluating-external-predictions)
- [Sweep Configuration](#sweep-configuration)
- [Output Files](#output-files)
- [Metrics](#metrics)
- [LLM Judge](#llm-judge)
- [Domain Support](#domain-support)
- [Useful Runner Options](#useful-runner-options)
- [Extending GateMem](#extending-gatemem)
- [Notes on Synthetic Data](#notes-on-synthetic-data)

---

## Overview

GateMem evaluates three dimensions:

- **Utility**: answer legitimate, authorized requests using current in-scope memory.
- **Access Control**: refuse or redact information when the requester is unauthorized or over-scoped.
- **Active Forgetting**: do not recover, confirm, or reconstruct information after explicit deletion.

The current release includes four domains:

- `medical`
- `office`
- `education`
- `household`

All benchmark data is synthetic.

---

## Important Terminology Note

The paper uses the terms:

```text
Utility
Access Control
Active Forgetting
````

The implementation keeps legacy internal names for compatibility with the original runs:

| Paper term        | Code / data `query_type` |
| ----------------- | ------------------------ |
| Utility           | `utility`                |
| Access Control    | `privacy`                |
| Active Forgetting | `safety`                 |

This also affects metric names:

| Paper metric                  | Code / summary field       |
| ----------------------------- | -------------------------- |
| Utility `U`                   | `utility_accuracy`         |
| Access-Control Violation `A`  | `privacy_leakage_rate`     |
| Active-Forgetting Failure `F` | `deletion_leakage_rate`    |
| Memory Governance Score `MGS` | `compliance_utility_score` |
| Over-Refusal `OR`             | `over_refusal_rate`        |

These names are kept to preserve compatibility with the original implementation and reported experimental runs.

---

## Directory Structure

```text
bench/
  agents/                 # Baseline and user-defined agent implementations
  data/                   # Domain datasets
    medical/
    office/
    education/
    household/
  eval/                   # Scoring, judging, and validation utilities
  prompts/                # Query and judge prompts
  scripts/
    run_eval.py           # Main evaluation driver
    score_predictions.py  # Score externally generated predictions
```

Other relevant top-level directories:

```text
configs/
  runs/                   # Single-run configs
  sweeps/                 # Sweep configs

docs/                     # User-facing benchmark documentation
outputs/
  <run_name>/             # Run outputs
scripts/
  sweep.py                # Sweep launcher
third_party/
  mem0_upstream/          # Optional vendored Mem0 upstream backend
```

---

## Data Format

Each domain directory contains:

```text
episodes.jsonl
checkpoints.jsonl
```

For example:

```text
bench/data/medical/episodes.jsonl
bench/data/medical/checkpoints.jsonl
```

### `episodes.jsonl`

Each line is one episode.

Important fields include:

* `episode_id`: unique episode ID
* `domain`: domain name
* `entities.principals`: principals in the episode

  * `principal_id`
  * `role`
  * `display_name`
* `entities.relationships`: relationship and access-relevant metadata
* `turns`: temporally ordered interaction turns

  * `turn_id`
  * `speaker`
  * `text`
  * optional timestamp or turn metadata
  * optional structured memory operations, such as deletion requests

Episodes contain long-form multi-party interactions where facts, permissions, and deletion requests evolve over time.

### `checkpoints.jsonl`

Each line is one hidden evaluation checkpoint.

Important fields include:

* `checkpoint_id`
* `episode_id`
* `as_of_turn_id`
* `asker`: authenticated requester identity and role
* `query_text`: user query
* `query_type`

  * `utility`
  * `privacy` for Access Control
  * `safety` for Active Forgetting
* `attack_type`: attack or failure-mode category for `privacy` and `safety`
* `expected_action`

  * `answer`
  * `answer_redacted`
  * `refuse`
  * `no_memory`
* `judge_spec`: authoritative grading specification
* `leak_targets`: protected strings or patterns for leakage evaluation

During benchmark evaluation, methods should **not** use `query_type`, `attack_type`, `expected_action`, `judge_spec`, or `leak_targets` as input. These fields are provided only for scoring, auditing, and analysis.

---

## Evaluation Protocol

Each episode is processed incrementally:

1. Reset the agent at the start of the episode.
2. Ingest turns in chronological order.
3. At each checkpoint boundary, ask the checkpoint query.
4. Record the agent response.
5. Continue processing the remaining turns.

This ensures that the agent can only use memory available up to the checkpoint turn.

See [`../docs/evaluation_protocol.md`](../docs/evaluation_protocol.md) for a compact protocol summary.

---

## Quickstart

Run Long-Context on the medical domain:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent long_context
```

Run Naive RAG:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent rag_naive
```

Run Policy RAG:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent rag_policy
```

Run with a config file:

```bash
python bench/scripts/run_eval.py \
  --config configs/runs/paper_main.yaml \
  --agent rag_policy \
  --run_name demo
```

---

## Running with Hosted LLM APIs

Set API keys as needed:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

OpenAI example with `gpt-4o-mini` as the answer model and `gpt-4o` as the judge:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent rag_policy \
  --llm_provider openai \
  --llm_model gpt-4o-mini \
  --temperature 0.2 \
  --max_output_tokens 4096 \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o \
  --judge_concurrency 4
```

Gemini example:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent rag_policy \
  --llm_provider gemini \
  --llm_model gemini-2.5-flash-lite \
  --temperature 0.2 \
  --max_output_tokens 4096 \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o
```

GPT-5 family example:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent rag_policy \
  --llm_provider openai \
  --llm_model gpt-5-mini \
  --reasoning_effort low \
  --text_verbosity low \
  --max_output_tokens 4096 \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o
```

---

## Baselines

### Baseline Summary

| Agent name     | Description                                                                      | Useful flags               |           |
| -------------- | -------------------------------------------------------------------------------- | -------------------------- | --------- |
| `long_context` | Uses available episode history directly in the prompt                            | none                       |           |
| `rag_naive`    | Retrieves prior memory chunks without explicit policy filtering                  | `--top_k`, embedding flags |           |
| `rag_policy`   | Retrieves memory with requester and policy awareness                             | `--top_k`, embedding flags |           |
| `a_mem`        | Agentic memory with metadata extraction, linking, graph expansion, and reranking | embedding flags            |           |
| `mem0`         | Mem0 baseline                                                                    | `--mem0_backend builtin    | upstream` |
| `remem`        | ReMeM baseline                                                                   | `--remem_variant iterative | single`   |

### Long-Context

Uses available episode history directly in the prompt.

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent long_context
```

### Naive RAG

Retrieves prior memory chunks without explicit policy filtering.

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent rag_naive \
  --retrieval_backend embedding \
  --embedding_impl native \
  --embed_provider openai \
  --embed_model text-embedding-3-small \
  --top_k 20
```

### Policy RAG

Retrieves memory with requester and policy awareness.

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent rag_policy \
  --retrieval_backend embedding \
  --embedding_impl native \
  --embed_provider openai \
  --embed_model text-embedding-3-small \
  --top_k 20
```

### A-Mem

Agentic memory with metadata extraction, linking, graph expansion, and reranking.

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent a_mem \
  --retrieval_backend embedding \
  --embedding_impl native \
  --embed_provider openai \
  --embed_model text-embedding-3-small \
  --top_k 20
```

### Mem0

Builtin backend:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent mem0 \
  --mem0_backend builtin
```

Upstream backend:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent mem0 \
  --mem0_backend upstream
```

### ReMeM

Iterative variant, ReMeM-I:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent remem \
  --remem_variant iterative
```

Single-step variant, ReMeM-S:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent remem \
  --remem_variant single
```

### Example Agent

A minimal example agent is included for users implementing new methods:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent example \
  --run_name example_agent_demo
```

See [`../docs/adding_new_agent.md`](../docs/adding_new_agent.md).

---

## Evaluating External Predictions

If your method is implemented outside this repository, generate a predictions JSONL file and score it directly:

```bash
python bench/scripts/score_predictions.py \
  --data_dir bench/data/medical \
  --predictions outputs/my_method/predictions.jsonl \
  --out_dir outputs/my_method_eval \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o
```

See [`../docs/prediction_format.md`](../docs/prediction_format.md) for the expected prediction schema.

---

## Sweep Configuration

GateMem provides a matrix-style sweep interface for reproducible experiments.

The main sweep config is:

```text
configs/sweeps/paper_matrix.yaml
```

Example: run all baselines on the medical domain with GPT-4o-mini:

```bash
python scripts/sweep.py \
  --config configs/sweeps/paper_matrix.yaml \
  --domains medical \
  --models gpt4omini
```

Example: run Policy RAG and Naive RAG on the office domain with Gemini:

```bash
python scripts/sweep.py \
  --config configs/sweeps/paper_matrix.yaml \
  --domains office \
  --models gemini_flash_lite \
  --baselines rag_naive rag_policy
```

Example: inspect commands before running:

```bash
python scripts/sweep.py \
  --config configs/sweeps/paper_matrix.yaml \
  --domains medical \
  --models gpt4omini \
  --dry_run
```

The sweep script supports:

```text
--domains
--models
--baselines
--dry_run
--max_parallel_runs
--continue_on_error
```

For a single run, use `bench/scripts/run_eval.py`. For a batch of runs, use `scripts/sweep.py`.

---

## Output Files

Each run writes to:

```text
outputs/<run_name>/
```

Main files:

* `predictions.jsonl`: raw model responses, selected memory context, timing, and token metadata
* `scores.jsonl`: per-checkpoint scoring records
* `judge_scores.jsonl`: LLM-judge outputs, when `--use_llm_judge` is enabled
* `summary.json`: aggregate metrics, runtime, token usage, and judge summaries

---

## Metrics

### Action Accuracy

Whether the predicted normalized action matches the expected action.

### Utility `U`

Effective authorized utility. A utility checkpoint is counted as successful when the model answers with the expected action and covers the required answer content.

Code field:

```text
utility_accuracy
```

### Access-Control Violation `A`

The rate at which the model discloses protected information to unauthorized or over-scoped requesters.

Legacy code field:

```text
privacy_leakage_rate
```

### Active-Forgetting Failure `F`

The rate at which the model recovers, confirms, or reconstructs explicitly deleted information.

Code field:

```text
deletion_leakage_rate
```

### Over-Refusal `OR`

The rate at which legitimate utility requests fail to receive the expected `answer` action.

Code field:

```text
over_refusal_rate
```

### Memory Governance Score `MGS`

```text
MGS = U * (1 - A) * (1 - F)
```

Legacy code field:

```text
compliance_utility_score
```

When `--use_llm_judge` is enabled, the main reported metrics are computed from LLM-judge labels.

---

## LLM Judge

Enable LLM-as-a-judge scoring:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/medical \
  --agent rag_policy \
  --llm_provider openai \
  --llm_model gpt-4o-mini \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o
```

The judge evaluates:

* action compliance
* required content coverage for utility checkpoints
* unauthorized disclosure for access-control checkpoints
* deleted-information recovery or confirmation for active-forgetting checkpoints

The paper experiments use `gpt-4o` as the judge model.

---

## Domain Support

The pipeline is domain-aware. The active domain is inferred from the dataset and used for:

* domain-specific access-policy prompt rendering
* domain-specific attack taxonomy validation
* domain-aware owner bucketing in Mem0
* domain-aware heuristics in A-Mem and retrieval baselines

Supported domains:

```text
medical
office
education
household
```

To run another domain, change `--data_dir`, for example:

```bash
python bench/scripts/run_eval.py \
  --data_dir bench/data/office \
  --agent rag_policy
```

---

## Useful Runner Options

```text
--run_name NAME                 Output directory name
--out_dir DIR                   Output root
--resume                        Resume an existing run
--no_progress                   Disable progress bar
--show_ingest_progress          Show turn-level ingestion progress
--episode_concurrency K         Parallelize over episodes
--judge_concurrency K           Parallelize judge calls
--log_level DEBUG               Verbose logging
--log_file run.log              Write logs to a file
```

---

## Extending GateMem

You can extend GateMem by:

* adding new episodes and checkpoints
* adding new domains and role structures
* adding new access-control or active-forgetting attack types
* implementing new agents under `bench/agents/`
* scoring externally generated predictions with `bench/scripts/score_predictions.py`
* modifying prompts under `bench/prompts/`
* experimenting with retrieval depth via `--top_k`
* experimenting with chunking via `--chunk_mode turn|window|chars`

---

## Notes on Synthetic Data

All released data is synthetic and for benchmark research only. The episodes are designed to represent realistic shared-memory governance situations, but they do not contain real patient records, workplace records, student records, or household data.
