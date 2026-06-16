from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .prompts import (
    FACT_SYSTEM_PROMPT,
    GIST_SYSTEM_PROMPT,
    render_fact_user_prompt,
    render_gist_user_prompt,
)


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    return s.strip()


def _extract_json_object(s: str) -> Optional[str]:
    s = _strip_code_fences(s)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start : end + 1]


def _parse_json_dict(s: str) -> Dict[str, Any]:
    js = _extract_json_object(s)
    if js is None:
        return {}
    try:
        obj = json.loads(js)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        try:
            obj = ast.literal_eval(js)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip(" -\n\t")
    if not s:
        return ""
    if s[-1] not in ".!?":
        s += "."
    return s


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+|;\s+", text or "")
    out: List[str] = []
    for piece in raw:
        piece = _clean_sentence(piece)
        if piece:
            out.append(piece)
    return out


def _extract_entities_from_text(text: str, speaker_id: str) -> List[str]:
    vals = [speaker_id]
    vals.extend(re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", text or ""))
    vals.extend(re.findall(r"\b[a-z]+_[a-z0-9_]+\b", text or "", flags=re.IGNORECASE))
    vals.extend(re.findall(r"\b[A-Za-z]{2,}-\d{2,}\b", text or ""))
    vals.extend(re.findall(r"\b[A-Z]{2,}\d+[A-Z0-9-]*\b", text or ""))
    seen = set()
    out = []
    for v in vals:
        v = str(v).strip()
        if not v or v.lower() in {"i", "we", "the", "and"}:
            continue
        if v.lower() in seen:
            continue
        seen.add(v.lower())
        out.append(v)
    return out


class EpisodicExtractor:
    def __init__(self, *, llm_router: Optional[Any], logger: Any = None) -> None:
        self.llm_router = llm_router
        self.logger = logger

    @property
    def use_llm(self) -> bool:
        return self.llm_router is not None and getattr(self.llm_router, "provider", "stub") != "stub"

    def extract(self, *, timestamp: Optional[str], principal_id: str, role: str, text: str) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        gists = self.extract_gists(timestamp=timestamp, principal_id=principal_id, role=role, text=text)
        facts = self.extract_facts(timestamp=timestamp, principal_id=principal_id, role=role, text=text, gists=gists)
        entities = self.extract_entities(text=text, principal_id=principal_id, facts=facts)
        return gists, facts, entities

    def extract_gists(self, *, timestamp: Optional[str], principal_id: str, role: str, text: str) -> List[str]:
        if self.use_llm:
            user_prompt = render_gist_user_prompt(timestamp=timestamp, principal_id=principal_id, role=role, text=text)
            res = self.llm_router.complete_result(system_prompt=GIST_SYSTEM_PROMPT, user_prompt=user_prompt)
            obj = _parse_json_dict(res.text)
            gists = obj.get("gists") if isinstance(obj, dict) else None
            if isinstance(gists, list):
                out = [_clean_sentence(str(x)) for x in gists if str(x).strip()]
                out = [x for x in out if x]
                if out:
                    return out
            if self.logger:
                self.logger.warning("ReMem gist extraction parse failed; falling back to heuristic")
        return self._heuristic_gists(timestamp=timestamp, principal_id=principal_id, role=role, text=text)

    def extract_facts(
        self,
        *,
        timestamp: Optional[str],
        principal_id: str,
        role: str,
        text: str,
        gists: List[str],
    ) -> List[Dict[str, Any]]:
        if self.use_llm:
            user_prompt = render_fact_user_prompt(
                timestamp=timestamp,
                principal_id=principal_id,
                role=role,
                text=text,
                gists=gists,
            )
            res = self.llm_router.complete_result(system_prompt=FACT_SYSTEM_PROMPT, user_prompt=user_prompt)
            obj = _parse_json_dict(res.text)
            facts = obj.get("facts") if isinstance(obj, dict) else None
            if isinstance(facts, list):
                cleaned: List[Dict[str, Any]] = []
                for fact in facts:
                    if not isinstance(fact, dict):
                        continue
                    subj = str(fact.get("subject") or "").strip()
                    pred = str(fact.get("predicate") or "").strip()
                    objv = str(fact.get("object") or "").strip()
                    if not subj or not pred or not objv:
                        continue
                    quals = fact.get("qualifiers")
                    if not isinstance(quals, dict):
                        quals = {}
                    cleaned.append({"subject": subj, "predicate": pred, "object": objv, "qualifiers": quals})
                if cleaned:
                    return cleaned
            if self.logger:
                self.logger.warning("ReMem fact extraction parse failed; falling back to heuristic")
        return self._heuristic_facts(timestamp=timestamp, principal_id=principal_id, role=role, text=text, gists=gists)

    def extract_entities(self, *, text: str, principal_id: str, facts: List[Dict[str, Any]]) -> List[str]:
        entities = _extract_entities_from_text(text, principal_id)
        for fact in facts:
            for key in ("subject", "object"):
                val = str(fact.get(key) or "").strip()
                if val:
                    entities.append(val)
        seen = set()
        out = []
        for e in entities:
            k = e.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(e)
        return out

    def _heuristic_gists(self, *, timestamp: Optional[str], principal_id: str, role: str, text: str) -> List[str]:
        prefix = f"[{timestamp}] " if timestamp else ""
        sentences = _split_sentences(text)
        if not sentences:
            return [prefix + f"{principal_id} ({role}) said: {_clean_sentence(text)}"] if text.strip() else []
        return [prefix + s for s in sentences]

    def _heuristic_facts(
        self,
        *,
        timestamp: Optional[str],
        principal_id: str,
        role: str,
        text: str,
        gists: List[str],
    ) -> List[Dict[str, Any]]:
        qualifiers = {"record_time": timestamp} if timestamp else {}
        facts: List[Dict[str, Any]] = []
        for gist in gists[:4]:
            stripped = re.sub(r"^\[[^\]]+\]\s*", "", gist).strip()
            if not stripped:
                continue
            m = re.match(r"(?P<subj>[A-Z][A-Za-z0-9_-]+|[a-z]+_[a-z0-9_]+)\s+(?P<pred>[^.]{1,40}?)\s+(?P<obj>[^.]{2,120})[.]?$", stripped)
            if m:
                subj = m.group("subj").strip()
                pred = m.group("pred").strip()
                objv = m.group("obj").strip()
            else:
                subj = principal_id
                pred = "described"
                objv = stripped[:160]
            facts.append({"subject": subj, "predicate": pred, "object": objv, "qualifiers": dict(qualifiers)})
        if not facts and text.strip():
            facts.append(
                {
                    "subject": principal_id,
                    "predicate": "said",
                    "object": _clean_sentence(text)[:160],
                    "qualifiers": dict(qualifiers),
                }
            )
        return facts
