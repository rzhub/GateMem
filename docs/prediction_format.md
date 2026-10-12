# Prediction Format and External Scoring

Researchers can evaluate a method on GateMem without integrating it into the repository. In that case, generate a `predictions.jsonl` file and score it with `bench/scripts/score_predictions.py`.

## Required fields

Each line should be a JSON object with at least:

```json
{
  "checkpoint_id": "...",
  "action": "answer",
  "answer": "...",
  "answer_structured": {},
  "used_record_ids": []
}
```

`action` must be one of:

```text
answer
answer_redacted
refuse
no_memory
```

Alternatively, you may use the same schema produced by `run_eval.py`:

```json
{
  "checkpoint_id": "...",
  "output": {
    "action": "answer",
    "answer": "...",
    "answer_structured": {},
    "used_record_ids": []
  }
}
```

The evaluator uses `checkpoint_id` to join predictions with the hidden checkpoint annotations.

## Score external predictions

Rule-based auxiliary scoring only:

```bash
python bench/scripts/score_predictions.py \
  --data_dir bench/data/medical \
  --predictions outputs/my_method/predictions.jsonl \
  --out_dir outputs/my_method_eval
```

LLM-judge scoring:

```bash
python bench/scripts/score_predictions.py \
  --data_dir bench/data/medical \
  --predictions outputs/my_method/predictions.jsonl \
  --out_dir outputs/my_method_eval \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o \
  --judge_concurrency 4
```

The script writes:

```text
outputs/my_method_eval/predictions.normalized.jsonl
outputs/my_method_eval/scores.jsonl
outputs/my_method_eval/judge_scores.jsonl      # if --use_llm_judge is enabled
outputs/my_method_eval/summary.json
```

## Prepare a leaderboard submission

Leaderboard submissions must use the official LLM judge configuration. Run the evaluator locally with your own API credentials:

```bash
export OPENAI_API_KEY="..."

python bench/scripts/score_predictions.py \
  --data_dir bench/data/medical \
  --predictions outputs/my_method/predictions.jsonl \
  --out_dir outputs/my_method_eval \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o \
  --judge_temperature 0 \
  --judge_concurrency 4
```

Copy the original predictions into the output directory and create the upload package:

```bash
cp outputs/my_method/predictions.jsonl outputs/my_method_eval/predictions.jsonl

cd outputs/my_method_eval
zip submission.zip \
  predictions.jsonl \
  predictions.normalized.jsonl \
  scores.jsonl \
  judge_scores.jsonl \
  summary.json
```

Upload `submission.zip` to [GateMem-Submit](https://huggingface.co/spaces/Ray368/GateMem-Submit). Submit one package for each method, backbone, and domain combination.

Accepted submissions are initially labeled **Community**. The Space checks checkpoint coverage, judge metadata, and metric consistency, but does not call the LLM judge. A submission is labeled **Verified** only after the maintainers independently re-score the exact same `predictions.jsonl`.

## Terminology note

The released data uses legacy internal names:

| Paper term | Code / data name |
|---|---|
| Utility | `utility` |
| Access Control | `privacy` |
| Active Forgetting | `safety` |

The scoring output preserves legacy field names for reproducibility. In paper terminology:

```text
privacy_leakage_rate  -> Access-Control Violation A
deletion_leakage_rate -> Active-Forgetting Failure F
compliance_utility_score -> Memory Governance Score MGS
```
