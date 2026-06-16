# GateMem Dataset Card

## Dataset summary

GateMem is a synthetic benchmark for evaluating memory governance in multi-principal shared-memory agents. It tests whether memory-augmented LLM agents can remain useful while enforcing access boundaries and honoring deletion requests.

## Domains

The dataset contains four domains:

- medical
- office
- education
- household

Each domain contains long-form multi-party episodes and hidden evaluation checkpoints.

## Files

Each domain directory contains:

```text
episodes.jsonl
checkpoints.jsonl
```

## Tasks

GateMem evaluates three dimensions:

- Utility
- Access Control
- Active Forgetting

The released data uses legacy `query_type` values for compatibility:

```text
utility -> Utility
privacy -> Access Control
safety  -> Active Forgetting
```

## Data generation

Episodes are synthetically generated and manually/structurally reviewed for consistency, evidence support, and leakage targets. The dataset does not contain real patient records, workplace records, student records, or household data.

## Intended use

GateMem is intended for research on memory-augmented agents, privacy-aware retrieval, access control, deletion behavior, and long-horizon memory evaluation.

## Out-of-scope use

GateMem should not be used as a source of real personal, medical, workplace, education, or household information. It should not be treated as a deployment-ready compliance test for any specific legal or institutional setting.

## Limitations

GateMem uses synthetic scenarios and LLM-based evaluation. It abstracts away real authentication systems, database-backed policy engines, and human-in-the-loop institutional review workflows.
