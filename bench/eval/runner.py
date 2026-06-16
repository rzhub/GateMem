from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Callable
import logging

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from bench.agents.base import Turn, Checkpoint, BaseMemoryAgent


def load_jsonl(path: str, *, ignore_errors: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                if ignore_errors:
                    continue
                raise
    return out


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def turn_from_raw(raw: Dict[str, Any]) -> Turn:
    speaker = raw.get("speaker", {}) or {}
    return Turn(
        turn_id=str(raw.get("turn_id")),
        timestamp=raw.get("timestamp"),
        speaker_principal_id=str(speaker.get("principal_id")),
        speaker_role=str(speaker.get("role")),
        text=str(raw.get("text", "")),
        record_refs=list(raw.get("record_refs", []) or []) or None,
        memory_ops=list(raw.get("memory_ops", []) or []) or None,
    )


def checkpoint_from_raw(raw: Dict[str, Any]) -> Checkpoint:
    asker = raw.get("asker", {}) or {}
    return Checkpoint(
        checkpoint_id=str(raw.get("checkpoint_id")),
        episode_id=str(raw.get("episode_id")),
        as_of_turn_id=str(raw.get("as_of_turn_id")),
        asker_principal_id=str(asker.get("principal_id")),
        asker_role=str(asker.get("role")),
        query_type=str(raw.get("query_type")),
        attack_type=(str(raw.get("attack_type")) if raw.get("attack_type") is not None else None),
        query_text=str(raw.get("query_text") or ""),
        expected_action=str(raw.get("expected_action") or ""),
        judge_spec=(raw.get("judge_spec") if isinstance(raw.get("judge_spec"), dict) else None),
        leak_targets=list(raw.get("leak_targets") or []) if isinstance(raw.get("leak_targets"), list) else None,
        # legacy (optional)
        gold_answer_structured=(raw.get("gold_answer_structured") if isinstance(raw.get("gold_answer_structured"), dict) else None),
        gold_refusal_category=(str(raw.get("gold_refusal_category")) if raw.get("gold_refusal_category") is not None else None),
        policy_snapshot=(raw.get("policy_snapshot") if isinstance(raw.get("policy_snapshot"), dict) else None),
    )


def run_episode(
    *,
    agent: BaseMemoryAgent,
    episode: Dict[str, Any],
    checkpoints: List[Dict[str, Any]],
    prediction_path: Optional[str] = None,
    pbar: Optional[Any] = None,
    append_lock: Optional[threading.Lock] = None,
    pbar_lock: Optional[threading.Lock] = None,
    logger: Optional[logging.Logger] = None,
    show_ingest_progress: bool = False,
) -> List[Dict[str, Any]]:
    logger = logger or logging.getLogger("bench")
    turns_raw = episode.get("turns", []) or []
    turns = [turn_from_raw(t) for t in turns_raw]

    turn_pos = {t.turn_id: i for i, t in enumerate(turns)}

    ckpts = [checkpoint_from_raw(c) for c in checkpoints]

    # sort by as_of_turn position
    ckpts.sort(key=lambda c: turn_pos.get(c.as_of_turn_id, -1))

    logger.info(
        "Episode=%s turns=%d checkpoints=%d",
        episode.get("episode_id"),
        len(turns),
        len(ckpts),
    )

    agent.reset(episode)
    logger.info("Agent reset complete. Starting incremental ingest/query...")

    ingested_upto = -1
    preds: List[Dict[str, Any]] = []

    for ckpt in ckpts:
        tgt = turn_pos.get(ckpt.as_of_turn_id)
        if tgt is None:
            continue

        # ingest incremental
        t_ingest0 = time.perf_counter()
        ingest_range = range(ingested_upto + 1, tgt + 1)
        if show_ingest_progress and (tgt - ingested_upto) > 1:
            it = tqdm(
                ingest_range,
                desc=f"Ingest {episode.get('episode_id')}->{ckpt.as_of_turn_id}",
                unit="turn",
                leave=False,
            )
        else:
            it = ingest_range

        n_new = 0
        for i in it:
            agent.ingest(turns[i])
            ingested_upto = i
            n_new += 1
        ingest_s = time.perf_counter() - t_ingest0

        logger.info(
            "Checkpoint=%s type=%s as_of=%s | ingested +%d turns (t=%0.2fs)",
            ckpt.checkpoint_id,
            ckpt.query_type,
            ckpt.as_of_turn_id,
            n_new,
            ingest_s,
        )

        t_query0 = time.perf_counter()
        logger.info(
            "Query start: asker=%s(%s) | text=%s",
            ckpt.asker_principal_id,
            ckpt.asker_role,
            (ckpt.query_text[:200] + " …") if len(ckpt.query_text) > 200 else ckpt.query_text,
        )
        output = agent.query(ckpt)
        query_s = time.perf_counter() - t_query0

        logger.info(
            "Query done: action=%s retrieval_s=%s llm_latency_s=%s total_query_s=%0.2fs",
            output.get("action"),
            output.get("retrieval_s"),
            output.get("llm_latency_s"),
            query_s,
        )

        timing = {
            "ingest_s": ingest_s,
            "query_s": query_s,
            "total_s": ingest_s + query_s,
        }
        preds.append(
            {
                "checkpoint_id": ckpt.checkpoint_id,
                "episode_id": ckpt.episode_id,
                "as_of_turn_id": ckpt.as_of_turn_id,
                "asker": {"principal_id": ckpt.asker_principal_id, "role": ckpt.asker_role},
                "query_type": ckpt.query_type,
                "attack_type": ckpt.attack_type,
                "expected_action": ckpt.expected_action,
                "query_text": ckpt.query_text,
                "output": output,
                "timing": timing,
            }
        )

        if prediction_path:
            if append_lock is None:
                _append_jsonl(prediction_path, preds[-1])
            else:
                with append_lock:
                    _append_jsonl(prediction_path, preds[-1])
        if pbar is not None:
            if pbar_lock is None:
                pbar.update(1)
            else:
                with pbar_lock:
                    pbar.update(1)

    return preds


def run_benchmark(
    *,
    agent: Optional[BaseMemoryAgent] = None,
    agent_factory: Optional[Callable[[], BaseMemoryAgent]] = None,
    episodes: List[Dict[str, Any]],
    checkpoints: List[Dict[str, Any]],
    prediction_path: Optional[str] = None,
    resume: bool = False,
    show_progress: bool = True,
    logger: Optional[logging.Logger] = None,
    show_ingest_progress: bool = False,
    episode_concurrency: int = 1,
) -> List[Dict[str, Any]]:
    logger = logger or logging.getLogger("bench")
    ckpts_by_episode: Dict[str, List[Dict[str, Any]]] = {}
    for c in checkpoints:
        eid = str(c.get("episode_id"))
        ckpts_by_episode.setdefault(eid, []).append(c)

    done_ids: Set[str] = set()
    if resume and prediction_path and os.path.exists(prediction_path):
        for row in load_jsonl(prediction_path, ignore_errors=True):
            cid = str(row.get("checkpoint_id"))
            if cid:
                done_ids.add(cid)

    total_remaining = 0
    for ep in episodes:
        eid = ep.get("episode_id")
        ep_ckpts = ckpts_by_episode.get(eid, [])
        total_remaining += sum(1 for c in ep_ckpts if str(c.get("checkpoint_id")) not in done_ids)

    pbar = tqdm(total=total_remaining, desc="Evaluating", unit="ckpt", disable=not show_progress)

    all_preds: List[Dict[str, Any]] = []

    # Serial execution (default)
    if int(episode_concurrency) <= 1:
        if agent is None:
            raise ValueError("run_benchmark: agent must be provided when episode_concurrency=1")
        for ep in episodes:
            eid = ep.get("episode_id")
            ep_ckpts = ckpts_by_episode.get(eid, [])
            if not ep_ckpts:
                continue

            # Filter already completed checkpoints for resume.
            if done_ids:
                ep_ckpts = [c for c in ep_ckpts if str(c.get("checkpoint_id")) not in done_ids]
                if not ep_ckpts:
                    continue

            preds = run_episode(
                agent=agent,
                episode=ep,
                checkpoints=ep_ckpts,
                prediction_path=prediction_path,
                pbar=pbar,
                logger=logger,
                show_ingest_progress=show_ingest_progress,
            )
            all_preds.extend(preds)
    else:
        # Parallel over episodes. Requires a fresh agent per episode.
        if agent_factory is None:
            raise ValueError("run_benchmark: agent_factory must be provided when episode_concurrency>1")

        append_lock = threading.Lock()
        pbar_lock = threading.Lock()

        def _run_one(ep: Dict[str, Any]) -> List[Dict[str, Any]]:
            eid = ep.get("episode_id")
            ep_ckpts = ckpts_by_episode.get(eid, [])
            if not ep_ckpts:
                return []

            if done_ids:
                ep_ckpts = [c for c in ep_ckpts if str(c.get("checkpoint_id")) not in done_ids]
                if not ep_ckpts:
                    return []

            local_agent = agent_factory()
            return run_episode(
                agent=local_agent,
                episode=ep,
                checkpoints=ep_ckpts,
                prediction_path=prediction_path,
                pbar=pbar,
                append_lock=append_lock,
                pbar_lock=pbar_lock,
                logger=logger,
                show_ingest_progress=show_ingest_progress,
            )

        # Submit episode tasks
        with ThreadPoolExecutor(max_workers=int(episode_concurrency)) as ex:
            futures = [ex.submit(_run_one, ep) for ep in episodes]
            for fut in as_completed(futures):
                try:
                    all_preds.extend(fut.result() or [])
                except Exception as e:
                    logger.exception("Episode worker failed: %s", e)
                    raise

    pbar.close()

    return all_preds
