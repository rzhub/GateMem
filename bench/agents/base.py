from __future__ import annotations

import json
import re

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple

from bench.domains import detect_domain_from_episode, format_relationship_fact, get_domain_label, get_query_policy_block


@dataclass
class Turn:
    turn_id: str
    speaker_principal_id: str
    speaker_role: str
    text: str
    timestamp: Optional[str] = None
    record_refs: Optional[List[str]] = None
    memory_ops: Optional[List[Dict[str, Any]]] = None


@dataclass
class Checkpoint:
    checkpoint_id: str
    episode_id: str
    as_of_turn_id: str
    asker_principal_id: str
    asker_role: str
    query_type: str
    # v2 schema fields (used for scoring/judging; NOT shown to the agent)
    attack_type: str | None = None
    query_text: str = ""
    expected_action: str = ""
    judge_spec: dict[str, Any] | None = None
    leak_targets: list[str] | None = None

    # Legacy fields (backward compatible; may be absent in v2 data)
    gold_answer_structured: dict[str, Any] | None = None
    gold_refusal_category: str | None = None
    policy_snapshot: dict[str, Any] | None = None


class MemoryAgent(Protocol):
    """Unified interface for memory agents."""

    def reset(self, episode: Dict[str, Any]) -> None:
        ...

    def ingest(self, turn: Turn) -> None:
        ...

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        ...


class BaseMemoryAgent:
    """Base class with shared prompt + LLM call helpers."""

    def __init__(
        self,
        *,
        top_k: int = 5,
        llm_mode: str = "leaky",  # used only in stub mode
        llm_router: Optional[Any] = None,
        query_prompt_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        answer_protocol: str = "standard",  # standard|native
    ) -> None:
        self.top_k = top_k
        self.llm_mode = llm_mode
        self.llm_router = llm_router
        self.episode: Optional[Dict[str, Any]] = None

        self.logger = logger or logging.getLogger("bench")
        protocol = str(answer_protocol or "standard").strip().lower()
        if protocol not in {"standard", "native"}:
            raise ValueError("answer_protocol must be standard|native")
        self.answer_protocol = protocol

        self.query_prompt_path = query_prompt_path
        self._query_template_cache: Optional[str] = None
        # Precompiled matchers to infer record_refs from text (internal only)
        self._record_matchers: List[Tuple[str, List[re.Pattern], List[str]]] = []

    def reset(self, episode: Dict[str, Any]) -> None:
        self.episode = episode
        self._record_matchers = self._compile_record_matchers(episode)
        self._query_template_cache = None

    def ingest(self, turn: Turn) -> None:
        raise NotImplementedError

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        raise NotImplementedError



    # ------------------------ Record ref inference (internal only) ------------------------

    @staticmethod
    def _compile_record_matchers(episode: Dict[str, Any]) -> List[Tuple[str, List[re.Pattern], List[str]]]:
        """Compile regex/canonical matchers for each record in the episode.

        Purpose:
        - Some turns/chunks may have missing or incomplete `record_refs` annotations.
        - For policy gating and auditability, we infer record references from chunk text
          using the dataset's own canonical values / regex patterns.
        - This inference is INTERNAL ONLY; inferred ids are never shown to the LLM.

        We intentionally keep canonical values conservative to reduce false positives.
        """
        matchers: List[Tuple[str, List[re.Pattern], List[str]]] = []
        records = (episode or {}).get("records") or []
        for rec in records:
            rid = rec.get("record_id")
            if not rid:
                continue

            # Compile regex list (ignore invalid patterns)
            compiled: List[re.Pattern] = []
            regex_list = rec.get("regex") or []
            if isinstance(regex_list, str):
                regex_list = [regex_list]
            for pat in regex_list:
                if not isinstance(pat, str) or not pat.strip():
                    continue
                try:
                    compiled.append(re.compile(pat, flags=re.IGNORECASE))
                except re.error:
                    continue

            # Keep only "distinctive" canonical strings to avoid matching generic words (e.g., "negative")
            canon_list = rec.get("canonical_values") or []
            if isinstance(canon_list, str):
                canon_list = [canon_list]
            canon_keep: List[str] = []
            for c in canon_list:
                if not isinstance(c, str) or not c.strip():
                    continue
                c2 = c.strip()
                has_digit = any(ch.isdigit() for ch in c2)
                has_sep = any(ch in c2 for ch in ["-", "_", "/"])
                if len(c2) >= 6 or has_digit or has_sep:
                    canon_keep.append(c2)

            matchers.append((str(rid), compiled, canon_keep))
        return matchers

    def _infer_record_refs(self, text: str) -> List[str]:
        """Infer record_ids referenced by `text` using compiled matchers (internal only)."""
        if not text:
            return []
        tl = text.lower()
        hits: List[str] = []
        for rid, patterns, canon in self._record_matchers:
            found = False
            for c in canon:
                if c.lower() in tl:
                    found = True
                    break
            if not found:
                for rgx in patterns:
                    if rgx.search(text):
                        found = True
                        break
            if found:
                hits.append(rid)
        return hits

    def _merge_record_refs(self, existing: Optional[List[str]], text: str) -> List[str]:
        """Union explicit `record_refs` with inferred ones, returning a stable sorted list."""
        out = set(existing or [])
        out.update(self._infer_record_refs(text))
        return sorted(out)


    # ------------------------ Prompt rendering ------------------------

    def _load_query_template(self) -> str:
        if self._query_template_cache is not None:
            return self._query_template_cache

        path = self.query_prompt_path
        if path is None:
            path = str((Path(__file__).resolve().parents[1] / "prompts" / "query_prompt.txt"))

        self._query_template_cache = Path(path).read_text(encoding="utf-8")
        return self._query_template_cache

    @staticmethod
    def _split_system_user(template: str) -> Tuple[str, str]:
        if "[SYSTEM]" in template and "[REQUEST CONTEXT]" in template:
            before, after = template.split("[REQUEST CONTEXT]", 1)
            system = before.replace("[SYSTEM]", "").strip()
            user = "[REQUEST CONTEXT]" + after
            return system.strip(), user.strip()
        return "", template


    @staticmethod
    def _relationship_mentions_principal(rel: Dict[str, Any], principal_id: str) -> bool:
        principal_id = str(principal_id or "").strip()
        if not principal_id:
            return False
        for key, value in (rel or {}).items():
            if not isinstance(value, str):
                continue
            key_low = str(key).lower()
            if key_low.endswith("_id") and value == principal_id:
                return True
        return False

    def _format_relationship_facts(self, checkpoint: Checkpoint) -> str:
        """Format requester-relevant relationship metadata.

        We intentionally avoid dumping the full episode relationship graph into the
        prompt. This keeps the prompt closer to a requester-centric, as-of-safe view
        and reduces leakage from unrelated relationship metadata.
        """
        if not self.episode:
            return "(none)"
        rels = (self.episode.get("entities", {}) or {}).get("relationships", []) or []
        if not rels:
            return "(none)"
        requester_id = str(checkpoint.asker_principal_id or "")
        filtered = [r for r in rels if isinstance(r, dict) and self._relationship_mentions_principal(r, requester_id)]
        if not filtered:
            return "(none)"
        return "\n".join(format_relationship_fact(r or {}) for r in filtered)

    @staticmethod
    def _format_retrieved_memory_block(retrieved: List[Dict[str, Any]]) -> str:
        if not retrieved:
            return "(none)"
        lines: List[str] = []
        for i, r in enumerate(retrieved, 1):
            speaker = str(r.get("principal_id") or r.get("speaker") or "unknown")
            txt = (r.get("text") or "").strip().replace("\n", " ")
            # if len(txt) > 800:
            #     txt = txt[:800] + " …"
            lines.append(f"Memory {i} (speaker={speaker}): {txt}")
        return "\n".join(lines)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Return a JSON-serializable copy of nested benchmark metadata.

        Retrieval traces are written to JSONL predictions. Some backends keep
        sets, tuples, numpy scalars, or other helper objects in metadata; those
        should not make prediction dumping fail. This helper deliberately keeps
        the original textual content unchanged, because context-exposure scoring
        must inspect exactly what the answer model could see.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(v) for v in value]
        if isinstance(value, set):
            return sorted(cls._json_safe(v) for v in value)
        try:
            import numpy as np  # type: ignore

            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        return str(value)

    @classmethod
    def _audit_memory_items(cls, retrieved_memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Canonicalize prompt-visible memory items for retrieval/privacy auditing."""
        items: List[Dict[str, Any]] = []
        for idx, item in enumerate(retrieved_memory or [], 1):
            if not isinstance(item, dict):
                items.append({"rank": idx, "text": str(item)})
                continue
            items.append(
                {
                    "rank": idx,
                    "record_id": cls._json_safe(item.get("record_id")),
                    "turn_id": cls._json_safe(item.get("turn_id")),
                    "principal_id": cls._json_safe(item.get("principal_id") or item.get("speaker")),
                    "role": cls._json_safe(item.get("role")),
                    "record_refs": cls._json_safe(item.get("record_refs") or []),
                    "score": cls._json_safe(item.get("score")),
                    "text": cls._json_safe(item.get("text") or ""),
                }
            )
        return items

    @classmethod
    def _build_memory_audit(
        cls,
        *,
        retrieved_memory: List[Dict[str, Any]],
        prompt_memory_block: str,
        context_format: str,
    ) -> Dict[str, Any]:
        """Build a trace of memory that was exposed to the answer model.

        Main invariant:
        - `prompt_context.text` is the exact memory block inserted into the
          answer prompt, after any policy filtering or native rendering.
        - `prompt_context.items` are the structured final memory items that
          produced that block when available.

        Scorers should use `prompt_context.text` as the source of truth for
        retrieval-stage / prompt-exposure leakage.
        """
        text = str(prompt_memory_block or "")
        return {
            "schema_version": 1,
            "stage": "prompt_context",
            "context_format": str(context_format or "standard"),
            "prompt_context": {
                "text": text,
                "n_chars": len(text),
                "n_items": len(retrieved_memory or []),
                "items": cls._audit_memory_items(retrieved_memory or []),
            },
        }

    def _run_llm(
        self,
        *,
        checkpoint: Checkpoint,
        retrieved_memory: List[Dict[str, Any]],
        rendered_memory_block_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call LLM (real provider or stub) using the unified prompt."""

        from .utils_llm import generate_llm_output

        template = self._load_query_template()
        system_tmpl, user_tmpl = self._split_system_user(template)

        retrieved_block = (
            rendered_memory_block_override
            if rendered_memory_block_override is not None
            else self._format_retrieved_memory_block(retrieved_memory)
        )
        domain_key = detect_domain_from_episode(self.episode)
        fmt_vars = {
            "asker_principal_id": checkpoint.asker_principal_id,
            "asker_role": checkpoint.asker_role,
            "relationship_facts_block": self._format_relationship_facts(checkpoint),
            "retrieved_memory_block": retrieved_block,
            "query_text": checkpoint.query_text,
            "domain_label": get_domain_label(domain_key),
            "global_access_policy_block": get_query_policy_block(domain_key),
        }


        try:
            system_prompt = system_tmpl.format(**fmt_vars)
            user_prompt = user_tmpl.format(**fmt_vars)
        except KeyError as e:
            missing = e.args[0] if e.args else "<unknown>"
            raise ValueError(
                f"Missing query prompt variable '{missing}' when formatting template."
            ) from e

        retrieved_text = "\n".join([m.get("text", "") for m in retrieved_memory])

        self.logger.debug(
            "LLM call: query_type=%s checkpoint_id=%s retrieved=%d",
            checkpoint.query_type,
            checkpoint.checkpoint_id,
            len(retrieved_memory),
        )
        out = generate_llm_output(
            query_text=checkpoint.query_text,
            retrieved_text=retrieved_text,
            query_type=checkpoint.query_type,
            expected_action=checkpoint.expected_action if self.llm_router is None or getattr(self.llm_router, "provider", "stub") == "stub" else "",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            llm_mode=self.llm_mode,
            llm_router=self.llm_router,
        )

        # Retrieval-stage privacy audit:
        # `retrieved_memory` remains the structured final memory items for
        # backward compatibility; `prompt_memory_block` is the exact memory text
        # inserted into the answer prompt and is the source of truth for
        # context-exposure scoring. This matters for "native" protocols where
        # the prompt-visible memory may include structured summaries, links, or
        # supporting evidence that are not present in the raw item text alone.
        context_format = "native" if rendered_memory_block_override is not None else "standard"
        out["retrieved_memory"] = self._json_safe(retrieved_memory)
        out["prompt_memory_block"] = retrieved_block
        out["memory_audit"] = self._build_memory_audit(
            retrieved_memory=retrieved_memory,
            prompt_memory_block=retrieved_block,
            context_format=context_format,
        )
        out["answer_protocol"] = self.answer_protocol
        return out
