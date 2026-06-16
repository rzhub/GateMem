# Reproducing Paper-Style Experiments

GateMem supports both single-run evaluation and matrix-style sweeps.

## Single run

```bash
python bench/scripts/run_eval.py \
  --config configs/runs/paper_main.yaml \
  --data_dir bench/data/medical \
  --agent rag_policy \
  --llm_provider openai \
  --llm_model gpt-4o-mini \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o \
  --run_name demo_medical_rag_policy
```

## Matrix sweep

The main sweep config is:

```text
configs/sweeps/paper_matrix.yaml
```

It defines four domains, three model backbones, and seven baselines.

Run all baselines on the medical domain with GPT-4o-mini:

```bash
python scripts/sweep.py \
  --config configs/sweeps/paper_matrix.yaml \
  --domains medical \
  --models gpt-4o-mini
```

Run selected baselines:

```bash
python scripts/sweep.py \
  --config configs/sweeps/paper_matrix.yaml \
  --domains medical \
  --models gpt-4o-mini \
  --baselines rag_naive rag_policy
```

Run office with Gemini:

```bash
python scripts/sweep.py \
  --config configs/sweeps/paper_matrix.yaml \
  --domains office \
  --models gemini-2.5-flash-lite
```

Print commands without running them:

```bash
python scripts/sweep.py \
  --config configs/sweeps/paper_matrix.yaml \
  --domains medical \
  --models gpt-4o-mini \
  --dry_run
```

## Available keys

Domains:

```text
medical, office, education, household
```

Model keys:

```text
gpt-4o-mini, gpt-5-mini, gemini-2.5-flash-lite
```

Baseline keys:

```text
long_context, rag_naive, rag_policy, a_mem, mem0, remem
```

## Outputs

Each run writes to:

```text
outputs/<run_name>/
```

Main files:

```text
predictions.jsonl
scores.jsonl
judge_scores.jsonl
summary.json
```

The main paper metrics are computed from LLM-judge labels when `--use_llm_judge` is enabled.
