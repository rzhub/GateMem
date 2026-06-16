#!/usr/bin/env python3
"""Score externally generated GateMem predictions.

This script is useful when a new method is implemented outside this repository.
It expects a predictions JSONL file with one row per checkpoint. Each row must
include `checkpoint_id` and either:

  1. an `output` object with action/answer fields, matching run_eval.py, or
  2. top-level `action`, `answer`, `answer_structured`, and `used_record_ids`.

Example:
  python bench/scripts/score_predictions.py \
    --data_dir bench/data/medical \
    --predictions outputs/my_method/predictions.jsonl \
    --out_dir outputs/my_method_eval \
    --use_llm_judge \
    --judge_provider openai \
    --judge_model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root on path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bench.eval.judge import run_llm_judge
from bench.eval.runner import dump_jsonl, load_jsonl
from bench.eval.scorer import score_predictions
from bench.eval.validator import validate_dataset
from bench.llm.router import LLMRouter
from bench.llm.types import LLMConfig


TOP_LEVEL_OUTPUT_KEYS = {"action", "answer", "answer_structured", "used_record_ids"}


def _normalize_prediction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize external prediction formats to the run_eval.py schema."""
    out = dict(row)
    if not out.get("checkpoint_id"):
        raise ValueError("Each prediction row must include checkpoint_id.")

    if isinstance(out.get("output"), dict):
        return out

    if any(k in out for k in TOP_LEVEL_OUTPUT_KEYS):
        output = {
            "action": out.get("action", ""),
            "answer": out.get("answer", ""),
            "answer_structured": out.get("answer_structured") or {},
            "used_record_ids": out.get("used_record_ids") or [],
        }
        # Preserve extra top-level model metadata under output.debug_external.
        extras = {
            k: v
            for k, v in out.items()
            if k not in TOP_LEVEL_OUTPUT_KEYS and k not in {"checkpoint_id"}
        }
        if extras:
            output["debug_external"] = extras
        out["output"] = output
        return out

    raise ValueError(
        "Prediction row must contain either an output object or top-level "
        "action/answer fields."
    )


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _promote_judge_metrics(summary: Dict[str, Any], judge_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Promote judge-derived metrics to the same top-level fields used by run_eval.py."""
    action_acc = float(judge_summary.get("judge_action_ok_rate") or 0.0)
    utility_acc = float(judge_summary.get("judge_effective_utility_accuracy") or 0.0)
    privacy_leak = float(judge_summary.get("judge_privacy_leakage_rate") or 0.0)
    deletion_leak = float(judge_summary.get("judge_deletion_leakage_rate") or 0.0)

    promoted = dict(summary)
    promoted.update(
        {
            "n_checkpoints": int(judge_summary.get("n_judged") or summary.get("n_checkpoints") or 0),
            "n_utility": int(judge_summary.get("n_utility") or summary.get("n_utility") or 0),
            "action_accuracy": action_acc,
            "utility_accuracy": utility_acc,
            "privacy_leakage_rate": privacy_leak,
            "deletion_leakage_rate": deletion_leak,
            "privacy_answer_leakage_rate": privacy_leak,
            "deletion_answer_leakage_rate": deletion_leak,
            "over_refusal_rate": float(summary.get("over_refusal_rate") or 0.0),
            "compliance_utility_score": utility_acc * (1.0 - privacy_leak) * (1.0 - deletion_leak),
            "llm_judge": judge_summary,
        }
    )
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser(description="Score externally generated GateMem predictions.")
    parser.add_argument("--data_dir", required=True, help="Domain data directory containing episodes.jsonl and checkpoints.jsonl.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl.")
    parser.add_argument("--out_dir", required=True, help="Directory where scores and summary will be written.")
    parser.add_argument("--gate_by_action", action="store_true", help="Apply post-hoc action gating to inner metrics.")

    parser.add_argument("--use_llm_judge", action="store_true", help="Run LLM-as-a-judge scoring.")
    parser.add_argument("--judge_provider", default="openai")
    parser.add_argument("--judge_model", default="gpt-4o")
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_max_output_tokens", type=int, default=4096)
    parser.add_argument("--judge_concurrency", type=int, default=4)
    parser.add_argument("--judge_prompt_path", default=None)
    parser.add_argument("--judge_api_base", default=None)
    parser.add_argument("--judge_api_key_env", default=None)
    parser.add_argument("--judge_reasoning_effort", default=None)
    parser.add_argument("--judge_text_verbosity", default=None)
    parser.add_argument("--merge_system_into_user", action="store_true")
    parser.add_argument("--resume_judge", action="store_true", help="Resume an existing judge_scores.jsonl if present.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    episodes_path = data_dir / "episodes.jsonl"
    checkpoints_path = data_dir / "checkpoints.jsonl"
    if not episodes_path.exists() or not checkpoints_path.exists():
        raise SystemExit(
            f"Invalid --data_dir={data_dir}. Expected episodes.jsonl and checkpoints.jsonl."
        )

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        raise SystemExit(f"Missing predictions file: {predictions_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_jsonl(str(episodes_path))
    checkpoints = load_jsonl(str(checkpoints_path))
    errors, warnings = validate_dataset(episodes=episodes, checkpoints=checkpoints, strict=True)
    for w in warnings:
        print(f"[WARN] DATA WARNING: {w}")
    if errors:
        for e in errors:
            print(f"[ERROR] DATA ERROR: {e}")
        raise SystemExit("Dataset validation failed.")

    raw_predictions = load_jsonl(str(predictions_path))
    predictions = [_normalize_prediction_row(r) for r in raw_predictions]

    normalized_pred_path = out_dir / "predictions.normalized.jsonl"
    dump_jsonl(str(normalized_pred_path), predictions)

    scores, summary = score_predictions(
        episodes=episodes,
        checkpoints=checkpoints,
        predictions=predictions,
        gate_by_action=bool(args.gate_by_action),
    )
    dump_jsonl(str(out_dir / "scores.jsonl"), scores)

    if args.use_llm_judge:
        prompt_path = args.judge_prompt_path or str(Path(__file__).resolve().parents[1] / "prompts" / "judge_prompt.txt")
        judge_cfg = LLMConfig(
            provider=args.judge_provider,
            model=args.judge_model,
            temperature=args.judge_temperature,
            max_output_tokens=args.judge_max_output_tokens,
            reasoning_effort=args.judge_reasoning_effort,
            text_verbosity=args.judge_text_verbosity,
            api_base=args.judge_api_base,
            api_key_env=args.judge_api_key_env,
            merge_system_into_user=args.merge_system_into_user,
        )
        judge_router = LLMRouter(judge_cfg)
        judge_rows, judge_summary = run_llm_judge(
            episodes=episodes,
            checkpoints=checkpoints,
            predictions=predictions,
            judge_router=judge_router,
            prompt_path=prompt_path,
            out_path=str(out_dir / "judge_scores.jsonl"),
            resume=bool(args.resume_judge),
            gate_by_action=bool(args.gate_by_action),
            concurrency=int(args.judge_concurrency),
        )
        summary = _promote_judge_metrics(summary, judge_summary)

    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
