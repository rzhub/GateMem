from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import time
import numpy as np

from bench.llm import LLMRouter
from bench.mem0 import Mem0Store, build_update_memory_prompt, build_user_fact_extraction_prompts
from bench.domains import detect_domain_from_episode

from .base import BaseMemoryAgent, Checkpoint, Turn


def _remove_code_blocks(content: str) -> str:
    """Match Mem0 behavior: remove enclosing ``` blocks and <think> traces."""
    content = (content or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if m:
        content = m.group(1).strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def _extract_json_str(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return (m.group(1) if m else text).strip()


def _parse_messages(messages: List[Dict[str, str]]) -> str:
    """Render messages as mem0-style transcript."""
    out: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "user", "assistant"):
            out.append(f"{role}: {content}")
    return "\n".join(out).strip() + "\n"


def _date_ymd_from_iso(ts: Optional[str]) -> str:
    if not ts:
        return "1970-01-01"
    if "T" in ts:
        return ts.split("T", 1)[0]
    return ts[:10]


@dataclass
class _BucketState:
    # Builtin backend state
    store: Mem0Store
    dialogue: List[Dict[str, str]]
    last_date_ymd: str


@dataclass
class _UpstreamBucketState:
    # Upstream backend state (dialogue buffer only)
    dialogue: List[Dict[str, str]]
    last_date_ymd: str


class Mem0Agent(BaseMemoryAgent):
    """Baseline: Mem0-style long-term memory.

    This benchmark provides two backends:

    1) builtin: self-contained Mem0 algorithm implementation (no external deps)
    2) upstream: vendored official Mem0 `Memory()` implementation (with its extract/update flow
       and SQLite history), configured to use an in-memory vector store adapter so it runs
       without external services.

    Benchmark-aligned design choices (both backends):
    - Memory scope: per episode (reset at episode boundary)
    - Owner buckets: domain-aware owner principal_id (stored as user_id)
    - Deletions: handled by Mem0's UPDATE/DELETE logic (no special-casing)
    - Retrieval for answering: search across all owner buckets in the episode (shared workspace)
      and return top_k memories with their owner principal_id in metadata.
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        llm_mode: str = "leaky",
        llm_router: Any = None,
        query_prompt_path: str | None = None,
        # builtin-only
        embed_router: Optional[Any] = None,
        # mem0 algorithm hyperparams
        mem0_backend: str = "builtin",
        mem0_message_window: int = 10,
        mem0_top_s: int = 5,
        mem0_max_facts: int = 20,
        # upstream settings
        mem0_cache_dir: Optional[str] = None,
        mem0_upstream_state_dir: Optional[str] = None,
        mem0_upstream_llm_provider: Optional[str] = None,
        mem0_upstream_llm_model: Optional[str] = None,
        mem0_upstream_openai_base: Optional[str] = None,
        mem0_upstream_api_key_env: Optional[str] = None,
        mem0_upstream_embed_provider: Optional[str] = None,
        mem0_upstream_embed_model: Optional[str] = None,
        mem0_upstream_embed_api_base: Optional[str] = None,
        mem0_upstream_embed_api_key_env: Optional[str] = None,
        mem0_upstream_embed_device: Optional[str] = None,
        mem0_llm_router: Optional[LLMRouter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            top_k=top_k,
            llm_mode=llm_mode,
            llm_router=llm_router,
            query_prompt_path=query_prompt_path,
            logger=logger,
        )

        backend = (mem0_backend or "builtin").lower().strip()
        if backend not in {"builtin", "upstream"}:
            raise ValueError(f"Unknown mem0_backend: {mem0_backend}")
        self.mem0_backend = backend

        self.mem0_message_window = int(mem0_message_window)
        self.mem0_top_s = int(mem0_top_s)
        self.mem0_max_facts = int(mem0_max_facts)

        # Builtin backend dependencies
        self._embed_router = embed_router

        # Memory-formation LLM for builtin backend
        self._mem_llm_router: Optional[LLMRouter] = mem0_llm_router or llm_router

        # Upstream settings
        self._mem0_cache_dir = mem0_cache_dir
        self._mem0_upstream_state_dir = mem0_upstream_state_dir
        self._mem0_upstream_llm_provider = (mem0_upstream_llm_provider or "openai").lower().strip()
        self._mem0_upstream_llm_model = mem0_upstream_llm_model
        self._mem0_upstream_openai_base = mem0_upstream_openai_base
        self._mem0_upstream_api_key_env = mem0_upstream_api_key_env
        self._mem0_upstream_embed_provider = (mem0_upstream_embed_provider or "openai").lower().strip()
        self._mem0_upstream_embed_model = mem0_upstream_embed_model
        self._mem0_upstream_embed_api_base = mem0_upstream_embed_api_base
        self._mem0_upstream_embed_api_key_env = mem0_upstream_embed_api_key_env
        self._mem0_upstream_embed_device = mem0_upstream_embed_device

        # Episode-scoped state
        self._owner_ids: set[str] = set()
        self._principal_roles: Dict[str, str] = {}
        self._active_owner_id: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._domain_key: str = "generic"

        # builtin buckets
        self._buckets: Dict[str, _BucketState] = {}
        # upstream buckets
        self._up_buckets: Dict[str, _UpstreamBucketState] = {}
        self._up_memory: Any = None  # upstream mem0.memory.main.Memory

        if self.mem0_backend == "builtin" and self._embed_router is None:
            raise ValueError("mem0_backend=builtin requires embed_router")

    # ----------------------------- lifecycle -----------------------------

    def reset(self, episode: Dict[str, Any]) -> None:
        super().reset(episode)
        self._episode_id = str(episode.get("episode_id") or "")
        self._domain_key = detect_domain_from_episode(episode)

        principals = (episode.get("entities", {}) or {}).get("principals", []) or []
        self._principal_roles = {
            str(p.get("principal_id")): str(p.get("role") or "")
            for p in principals
            if p.get("principal_id")
        }
        # Multi-principal adaptation of Mem0: every non-agent principal gets its own
        # memory scope. This generalizes the official two-speaker setup without
        # collapsing non-owner roles into auxiliary context.
        self._owner_ids = {
            str(p.get("principal_id"))
            for p in principals
            if p.get("principal_id") and str(p.get("role") or "") != "agent"
        }
        self._active_owner_id = None

        if self.mem0_backend == "builtin":
            self._buckets = {}
            for pid in sorted(self._owner_ids):
                self._buckets[str(pid)] = _BucketState(
                    store=Mem0Store(), dialogue=[], last_date_ymd="1970-01-01"
                )
            self._up_buckets = {}
            self._up_memory = None
            return

        # upstream
        self._buckets = {}
        self._up_buckets = {}
        for pid in sorted(self._owner_ids):
            self._up_buckets[str(pid)] = _UpstreamBucketState(dialogue=[], last_date_ymd="1970-01-01")

        self._init_upstream_memory(episode_id=self._episode_id)

    def ingest(self, turn: Turn) -> None:
        """Ingest one turn into every principal bucket.

        Multi-principal adaptation:
        - each non-agent principal has its own Mem0 scope
        - a speaker's own utterances are stored as `user` in their bucket
        - other principals see that same utterance as `assistant` context
        - agent/system turns are appended as `assistant` context to every bucket

        This preserves the Mem0 invariant that personal facts are extracted from the
        bucket owner's `user` turns while still retaining surrounding dialogue context.
        """
        speaker_id = str(turn.speaker_principal_id or "")
        is_owner_turn = speaker_id in self._owner_ids and turn.speaker_role != "agent"

        if is_owner_turn:
            self._active_owner_id = speaker_id
            for owner_id in sorted(self._owner_ids):
                role = "user" if owner_id == speaker_id else "assistant"
                self._append_dialogue(owner_id, role=role, text=turn.text, timestamp=turn.timestamp)
            self._mem0_update_bucket(speaker_id)
            return

        for owner_id in sorted(self._owner_ids):
            self._append_dialogue(owner_id, role="assistant", text=turn.text, timestamp=turn.timestamp)

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        t0 = time.perf_counter()
        retrieved_memory = self._retrieve_memories_global(checkpoint.query_text, top_k=self.top_k)
        out = self._run_llm(checkpoint=checkpoint, retrieved_memory=retrieved_memory)
        out["retrieval_s"] = time.perf_counter() - t0
        return out

    # ----------------------------- common helpers -----------------------------

    def _append_dialogue(self, owner_id: str, *, role: str, text: str, timestamp: Optional[str]) -> None:
        if self.mem0_backend == "builtin":
            b = self._buckets.get(owner_id)
            if not b:
                return
            b.dialogue.append({"role": role, "content": str(text or "")})
            b.last_date_ymd = _date_ymd_from_iso(timestamp)
            if len(b.dialogue) > max(2 * self.mem0_message_window, 50):
                b.dialogue = b.dialogue[-max(2 * self.mem0_message_window, 50) :]
            return

        b2 = self._up_buckets.get(owner_id)
        if not b2:
            return
        b2.dialogue.append({"role": role, "content": str(text or "")})
        b2.last_date_ymd = _date_ymd_from_iso(timestamp)
        if len(b2.dialogue) > max(2 * self.mem0_message_window, 50):
            b2.dialogue = b2.dialogue[-max(2 * self.mem0_message_window, 50) :]

    def _merge_bucket_hits_diversified(
        self,
        bucket_hits: Dict[str, List[Dict[str, Any]]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Round-robin merge per-bucket retrieval results.

        Official Mem0 retrieves memories per speaker scope. In GateMem's
        multi-principal setting, we preserve that spirit by searching each
        principal bucket independently and then interleaving results so one
        heavily populated principal cannot monopolize the final context.
        """
        if not bucket_hits:
            return []

        ordered_buckets = sorted(
            (
                pid
                for pid, hits in bucket_hits.items()
                if hits
            ),
            key=lambda pid: bucket_hits[pid][0].get("score", 0.0),
            reverse=True,
        )
        merged: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        depth = 0
        k = max(1, int(top_k))
        while len(merged) < k:
            added = False
            for pid in ordered_buckets:
                hits = bucket_hits.get(pid) or []
                if depth >= len(hits):
                    continue
                item = hits[depth]
                rid = str(item.get("record_id") or "")
                if rid and rid in seen_ids:
                    continue
                if rid:
                    seen_ids.add(rid)
                merged.append(item)
                added = True
                if len(merged) >= k:
                    break
            if not added:
                break
            depth += 1
        return merged

    # ----------------------------- builtin mem0 -----------------------------

    def _mem0_update_bucket_builtin(self, owner_id: str) -> None:
        b = self._buckets.get(owner_id)
        if not b:
            return

        if self._mem_llm_router is None or getattr(self._mem_llm_router, "provider", "stub") == "stub":
            return

        window = b.dialogue[-self.mem0_message_window :]
        transcript = _parse_messages(window)
        sys_prompt, user_prefix = build_user_fact_extraction_prompts(today_ymd=b.last_date_ymd)

        res = self._mem_llm_router.complete_result(system_prompt=sys_prompt, user_prompt=user_prefix + transcript)
        raw = _remove_code_blocks(res.text)

        facts: List[str] = []
        try:
            obj = json.loads(_extract_json_str(raw))
            if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
                facts = [
                    str(x).strip()
                    for x in obj.get("facts")
                    if isinstance(x, (str, int, float))
                ]
        except Exception:
            facts = []

        facts = [f for f in facts if f]
        if not facts:
            return
        if self.mem0_max_facts > 0:
            facts = facts[: self.mem0_max_facts]

        # Retrieve candidate old memories for update (top_s per fact), then dedupe.
        old_items: Dict[str, Dict[str, str]] = {}
        fact_embs, _ = self._embed_router.embed_texts(facts)  # type: ignore[union-attr]
        for fact, emb in zip(facts, fact_embs):
            for item, _score in b.store.search(query_embedding=emb, top_k=self.mem0_top_s):
                old_items[item.id] = {"id": item.id, "text": item.text}

        retrieved_old = list(old_items.values())

        # Remap IDs to short integers for the update prompt (matches upstream Mem0).
        temp_to_real: Dict[str, str] = {}
        retrieved_old_temp: List[Dict[str, str]] = []
        for idx, item in enumerate(retrieved_old):
            real_id = str(item.get("id"))
            tmp_id = str(idx)
            temp_to_real[tmp_id] = real_id
            retrieved_old_temp.append({"id": tmp_id, "text": str(item.get("text", ""))})

        update_prompt = build_update_memory_prompt(
            retrieved_old_memory=retrieved_old_temp, new_facts=facts
        )
        upd = self._mem_llm_router.complete_result(system_prompt="", user_prompt=update_prompt)
        upd_raw = _remove_code_blocks(upd.text)

        try:
            upd_obj = json.loads(_extract_json_str(upd_raw))
        except Exception:
            return

        actions = upd_obj.get("memory") if isinstance(upd_obj, dict) else None
        if not isinstance(actions, list):
            return

        # Apply actions.
        for a in actions:
            if not isinstance(a, dict):
                continue
            event = str(a.get("event", "")).strip().upper()
            mid = str(a.get("id", "")).strip()
            text = str(a.get("text", "") or "").strip()
            if not event or event == "NONE":
                continue

            if event in {"UPDATE", "DELETE"}:
                real_id = temp_to_real.get(mid)
                if not real_id:
                    continue
                if event == "DELETE":
                    b.store.delete(item_id=real_id)
                    continue
                if not text:
                    continue
                emb, _ = self._embed_router.embed_texts([text])  # type: ignore[union-attr]
                b.store.update(item_id=real_id, text=text, embedding=emb[0])
                continue

            if event == "ADD":
                if not text:
                    continue
                emb, _ = self._embed_router.embed_texts([text])  # type: ignore[union-attr]
                suggested = mid if mid and mid not in temp_to_real else None
                b.store.add(text=text, embedding=emb[0], suggested_id=suggested)

    def _retrieve_memories_global_builtin(self, query_text: str, *, top_k: int) -> List[Dict[str, Any]]:
        if not query_text:
            return []
        q_emb, _ = self._embed_router.embed_query(query_text)  # type: ignore[union-attr]
        per_bucket: Dict[str, List[Dict[str, Any]]] = {}
        for pid, b in self._buckets.items():
            hits: List[Dict[str, Any]] = []
            for item, score in b.store.search(query_embedding=q_emb, top_k=top_k):
                hits.append(
                    {
                        "record_id": f"mem0:{pid}:{item.id}",
                        "turn_id": "",
                        "principal_id": pid,
                        "role": self._principal_roles.get(pid, "owner"),
                        "text": item.text,
                        "record_refs": [],
                        "score": float(score),
                    }
                )
            if hits:
                per_bucket[pid] = hits
        return self._merge_bucket_hits_diversified(per_bucket, top_k=top_k)

    # ----------------------------- upstream mem0 -----------------------------

    @staticmethod
    def _normalize_mem0_upstream_llm_provider(provider: Optional[str]) -> str:
        p = (provider or "openai").lower().strip()
        if p in {"openai", "gemini", "deepseek"}:
            return p
        # Hosted Llama endpoints are OpenAI-compatible in this benchmark.
        if p in {"llama", "nvidia"}:
            return "openai"
        raise ValueError(
            "mem0 upstream currently supports llm_provider=openai, gemini, deepseek, or llama/nvidia. "
            f"Received: {provider!r}."
        )

    @staticmethod
    def _normalize_mem0_upstream_embed_provider(provider: Optional[str]) -> str:
        p = (provider or "openai").lower().strip()
        if p == "hf":
            return "huggingface"
        if p in {"openai", "gemini", "huggingface"}:
            return p
        raise ValueError(
            "mem0 upstream currently supports embed_provider=openai, gemini, or hf/huggingface. "
            f"Received: {provider!r}."
        )

    @staticmethod
    def _read_api_key_from_env(env_name: Optional[str]) -> Optional[str]:
        if not env_name:
            return None
        v = os.getenv(env_name)
        return v if v else None

    def _init_upstream_memory(self, *, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id is required for mem0 upstream")

        # Prepare vendored Mem0 upstream and install shims (without shadowing user's packages).
        root = Path(__file__).resolve().parents[2]
        from bench.integrations.mem0_upstream import ensure_upstream_ready

        ensure_upstream_ready(repo_root=str(root), state_dir=self._mem0_upstream_state_dir)

        raw_llm_provider = (self._mem0_upstream_llm_provider or "openai").lower().strip()
        llm_provider = self._normalize_mem0_upstream_llm_provider(raw_llm_provider)
        embed_provider = self._normalize_mem0_upstream_embed_provider(self._mem0_upstream_embed_provider)

        llm_api_key_env = self._mem0_upstream_api_key_env
        if not llm_api_key_env:
            if raw_llm_provider in {"llama", "nvidia"}:
                llm_api_key_env = "NVIDIA_API_KEY"
            elif raw_llm_provider == "deepseek":
                llm_api_key_env = "DEEPSEEK_API_KEY"
            elif llm_provider == "gemini":
                llm_api_key_env = "GEMINI_API_KEY"
            elif llm_provider == "openai":
                llm_api_key_env = "OPENAI_API_KEY"
        llm_api_key = self._read_api_key_from_env(llm_api_key_env)

        embed_api_key_env = self._mem0_upstream_embed_api_key_env
        if not embed_api_key_env:
            if embed_provider == "openai":
                embed_api_key_env = "OPENAI_API_KEY"
            elif embed_provider == "gemini":
                embed_api_key_env = "GEMINI_API_KEY"
        embed_api_key = self._read_api_key_from_env(embed_api_key_env)

        llm_openai_base = self._mem0_upstream_openai_base
        if raw_llm_provider in {"llama", "nvidia"} and not llm_openai_base:
            llm_openai_base = "https://integrate.api.nvidia.com/v1"

        embed_openai_base = self._mem0_upstream_embed_api_base
        if embed_provider == "openai" and not embed_openai_base:
            embed_openai_base = self._mem0_upstream_openai_base if raw_llm_provider == "openai" else "https://api.openai.com/v1"

        # Gemini upstream components read GOOGLE_API_KEY unless api_key is set in config.
        if (llm_provider == "gemini" or embed_provider == "gemini") and (llm_api_key or embed_api_key):
            os.environ.setdefault("GOOGLE_API_KEY", llm_api_key or embed_api_key or "")

        os.environ["GATEMEM_MEM0_TOP_S"] = str(max(1, int(self.mem0_top_s)))

        # Prepare episode-scoped storage path (history DB lives here).
        base = self._mem0_cache_dir or str(root / "outputs" / "mem0_cache")
        ep_dir = os.path.join(base, "upstream", episode_id)
        os.makedirs(ep_dir, exist_ok=True)

        from mem0.configs.base import MemoryConfig
        from mem0.memory.main import Memory
        from mem0.vector_stores.configs import VectorStoreConfig

        vs_cfg = {
            "collection_name": f"mem0_{episode_id}",
            "distance_strategy": "cosine",
            "embedding_model_dims": 1536,
        }
        vs = VectorStoreConfig(provider="in_memory", config=vs_cfg)

        cfg = MemoryConfig(vector_store=vs)
        cfg.history_db_path = os.path.join(ep_dir, "history.db")

        # LLM / embedder configuration (provider-aware)
        cfg.llm.provider = llm_provider
        default_llm_model = (
            "gemini-2.5-flash-lite"
            if llm_provider == "gemini"
            else "deepseek-v4-pro"
            if llm_provider == "deepseek"
            else "meta/llama-4-maverick-17b-128e-instruct"
            if raw_llm_provider in {"llama", "nvidia"}
            else "gpt-4o-mini"
        )
        cfg.llm.config = {
            "model": self._mem0_upstream_llm_model or default_llm_model,
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        if llm_api_key:
            cfg.llm.config["api_key"] = llm_api_key
        if llm_provider == "openai" and llm_openai_base:
            cfg.llm.config["openai_base_url"] = llm_openai_base
        if llm_provider == "deepseek" and self._mem0_upstream_openai_base:
            cfg.llm.config["deepseek_base_url"] = self._mem0_upstream_openai_base

        cfg.embedder.provider = embed_provider
        cfg.embedder.config = {
            "model": self._mem0_upstream_embed_model
            or ("text-embedding-3-small" if embed_provider == "openai" else ("models/text-embedding-004" if embed_provider == "gemini" else "sentence-transformers/all-MiniLM-L6-v2")),
            "embedding_dims": 1536 if embed_provider == "openai" else (768 if embed_provider == "gemini" else None),
        }
        if embed_api_key and embed_provider in {"openai", "gemini"}:
            cfg.embedder.config["api_key"] = embed_api_key
        if embed_provider == "openai":
            cfg.embedder.config["openai_base_url"] = embed_openai_base
        elif embed_provider == "huggingface":
            cfg.embedder.config["model_kwargs"] = {"device": self._mem0_upstream_embed_device or "cpu"}
        elif embed_provider == "gemini":
            cfg.embedder.config["output_dimensionality"] = 768

        # Drop None values so upstream config constructors stay happy.
        cfg.embedder.config = {k: v for k, v in cfg.embedder.config.items() if v is not None}

        # Disable graph store by default.
        cfg.graph_store.config = None

        self._up_memory = Memory(cfg)

    def _mem0_update_bucket_upstream(self, owner_id: str) -> None:
        b = self._up_buckets.get(owner_id)
        if not b or self._up_memory is None:
            return

        # In stub mode, skip memory formation.
        # (Upstream mem0 uses its own provider stack; we treat provider=stub as "offline".)
        if getattr(self.llm_router, "provider", "stub") == "stub":
            return

        window = b.dialogue[-self.mem0_message_window :]
        if not window:
            return

        # Scope: user_id=owner_id, run_id=episode_id (episode scope)
        try:
            self._up_memory.add(
                window,
                user_id=str(owner_id),
                run_id=str(self._episode_id),
                metadata={"actor_id": str(owner_id)},
                infer=True,
            )
        except Exception as e:
            # Fail-soft: Mem0 upstream errors shouldn't crash the benchmark run.
            self.logger.warning("mem0 upstream add failed: %s", e)

    def _retrieve_memories_global_upstream(self, query_text: str, *, top_k: int) -> List[Dict[str, Any]]:
        if not query_text or self._up_memory is None:
            return []

        per_bucket: Dict[str, List[Dict[str, Any]]] = {}
        for owner_id in sorted(self._owner_ids):
            try:
                res = self._up_memory.search(
                    query_text,
                    user_id=str(owner_id),
                    run_id=str(self._episode_id),
                    limit=max(1, int(top_k)),
                    rerank=False,
                )
            except Exception as e:
                self.logger.warning("mem0 upstream search failed for %s: %s", owner_id, e)
                continue

            items = (res or {}).get("results") if isinstance(res, dict) else None
            if not isinstance(items, list):
                continue

            hits: List[Dict[str, Any]] = []
            for it in items[: max(1, int(top_k))]:
                if not isinstance(it, dict):
                    continue
                mid = str(it.get("id") or "")
                mem_txt = str(it.get("memory") or "").strip()
                owner = str(it.get("user_id") or owner_id)
                score = float(it.get("score") or 0.0)
                if not mem_txt:
                    continue
                hits.append(
                    {
                        "record_id": f"mem0_up:{owner}:{mid}",
                        "turn_id": "",
                        "principal_id": owner,
                        "role": self._principal_roles.get(owner, "owner"),
                        "text": mem_txt,
                        "record_refs": [],
                        "score": score,
                    }
                )
            if hits:
                per_bucket[owner_id] = hits
        return self._merge_bucket_hits_diversified(per_bucket, top_k=top_k)

    # ----------------------------- dispatch -----------------------------

    def _mem0_update_bucket(self, owner_id: str) -> None:
        if self.mem0_backend == "builtin":
            self._mem0_update_bucket_builtin(owner_id)
        else:
            self._mem0_update_bucket_upstream(owner_id)

    def _retrieve_memories_global(self, query_text: str, *, top_k: int) -> List[Dict[str, Any]]:
        if self.mem0_backend == "builtin":
            return self._retrieve_memories_global_builtin(query_text, top_k=top_k)
        return self._retrieve_memories_global_upstream(query_text, top_k=top_k)
