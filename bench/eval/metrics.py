from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _norm_str(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def canonicalize(obj: Any) -> Any:
    """Canonicalize JSON-like structures for stable comparison."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return _norm_str(obj)
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, list):
        canon = [canonicalize(x) for x in obj]
        try:
            return sorted(canon)
        except TypeError:
            return sorted(canon, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj.keys())}
    return str(obj)


def structured_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    return canonicalize(pred) == canonicalize(gold)


def action_match(pred_action: str, expected_action: str) -> bool:
    pred_action = (pred_action or "").strip()
    expected_action = (expected_action or "").strip()
    return pred_action == expected_action


@dataclass
class RecordMatcher:
    regex: List[str]
    canonical_values: List[str]


def _text_for_leak_scan(pred_output: Dict[str, Any]) -> str:
    answer = str(pred_output.get("answer") or "")
    answer_structured = pred_output.get("answer_structured") or {}
    try:
        struct_text = json.dumps(answer_structured, ensure_ascii=False, sort_keys=True)
    except Exception:
        struct_text = str(answer_structured)
    return answer + "\n" + struct_text


def _matcher_hits_text(text: str, m: RecordMatcher) -> bool:
    low = text.lower()
    for v in m.canonical_values:
        if not v:
            continue
        vv = v.lower()
        if len(vv) < 4:
            continue
        if vv in low:
            return True
    for pat in m.regex:
        if not pat:
            continue
        try:
            if re.search(pat, text, flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def infer_required_record_ids_from_gold(
    gold_struct: Dict[str, Any],
    record_matchers: Dict[str, RecordMatcher],
) -> List[str]:
    """Heuristic inference of which record_ids are needed to produce the gold answer."""

    def _is_scalar(x: Any) -> bool:
        return x is None or isinstance(x, (str, int, float, bool))

    def _leaf_texts(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, bool):
            return ["true" if x else "false"]
        if isinstance(x, (str, int, float)):
            s = str(x).strip()
            return [s] if s else []
        if isinstance(x, dict):
            out: List[str] = []
            for v in x.values():
                out.extend(_leaf_texts(v))
            return out
        if isinstance(x, list):
            out = []
            for it in x:
                out.extend(_leaf_texts(it))
            return out
        s = str(x).strip()
        return [s] if s else []

    def _collect_fragments(x: Any) -> List[str]:
        frags: List[str] = []
        if _is_scalar(x):
            frags.extend(_leaf_texts(x))
            return frags
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, (str, int, float)) and v is not None:
                    kv = f"{k} {v}".strip()
                    if kv:
                        frags.append(kv)
            leaves = _leaf_texts(x)
            if len(leaves) >= 2:
                frags.append(" ".join(leaves))
            for v in x.values():
                frags.extend(_collect_fragments(v))
            return frags
        if isinstance(x, list):
            for it in x:
                frags.extend(_collect_fragments(it))
            return frags
        frags.extend(_leaf_texts(x))
        return frags

    pieces: List[str] = []
    try:
        pieces.append(json.dumps(gold_struct, ensure_ascii=False, sort_keys=True))
    except Exception:
        pieces.append(str(gold_struct))
    pieces.extend(_collect_fragments(gold_struct))

    seen = set()
    normed: List[str] = []
    for p in pieces:
        if not p:
            continue
        pp = re.sub(r"\s+", " ", str(p)).strip()
        if not pp or pp in seen:
            continue
        seen.add(pp)
        normed.append(pp)
    gold_text = "\n".join(normed)
    req: List[str] = []
    for rid, m in record_matchers.items():
        if _matcher_hits_text(gold_text, m):
            req.append(rid)
    return sorted(set(req))


def utility_match(
    pred_output: Dict[str, Any],
    gold_struct: Dict[str, Any],
    record_matchers: Dict[str, RecordMatcher],
) -> bool:
    pred_struct = pred_output.get("answer_structured") or {}
    if isinstance(pred_struct, dict) and structured_match(pred_struct, gold_struct):
        return True

    req_ids = infer_required_record_ids_from_gold(gold_struct, record_matchers)
    if not req_ids:
        return isinstance(pred_struct, dict) and structured_match(pred_struct, gold_struct)

    text = _text_for_leak_scan(pred_output)
    for rid in req_ids:
        m = record_matchers.get(rid)
        if not m:
            continue
        if not _matcher_hits_text(text, m):
            return False
    return True


# ------------------------- v2 rule-based helpers -------------------------

def flatten_output_text(pred_output: Dict[str, Any]) -> str:
    answer = str(pred_output.get("answer", "") or "")
    ans_struct = pred_output.get("answer_structured", {}) or {}
    try:
        ans_struct_s = json.dumps(ans_struct, ensure_ascii=False, sort_keys=True)
    except Exception:
        ans_struct_s = str(ans_struct)
    return answer + "\n" + ans_struct_s


def _jsonish_text(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def _append_unique_text(pieces: List[str], seen: set[str], value: Any) -> None:
    if value is None:
        return
    s = str(value)
    if not s:
        return
    # Deduplicate exact repeated blocks while preserving order.
    if s in seen:
        return
    seen.add(s)
    pieces.append(s)


def flatten_prompt_context_text(pred_output: Dict[str, Any]) -> str:
    """Flatten the memory context exposed to the answer model.

    Source-of-truth for new predictions:
    - output.memory_audit.prompt_context.text

    Backward-compatible fallbacks:
    - output.prompt_memory_block
    - output.retrieved_memory[*].text
    - native/debug structures when present

    This function intentionally does NOT scan the final answer. It is used for
    retrieval-stage / prompt-exposure leakage metrics.
    """
    pieces: List[str] = []
    seen: set[str] = set()

    audit = pred_output.get("memory_audit")
    if isinstance(audit, dict):
        prompt_context = audit.get("prompt_context")
        if isinstance(prompt_context, dict):
            _append_unique_text(pieces, seen, prompt_context.get("text"))

    _append_unique_text(pieces, seen, pred_output.get("prompt_memory_block"))

    # Fallback for predictions produced before memory_audit existed.
    retrieved = pred_output.get("retrieved_memory")
    if isinstance(retrieved, list):
        for item in retrieved:
            if isinstance(item, dict):
                _append_unique_text(pieces, seen, item.get("text"))
            else:
                _append_unique_text(pieces, seen, item)

    # Native protocol fallbacks. These may include prompt-visible summaries,
    # gists, facts, entities, or links beyond raw retrieved_memory text.
    for key in ("amem_native_cards", "remem_evidence_packets"):
        val = pred_output.get(key)
        if val:
            _append_unique_text(pieces, seen, _jsonish_text(val))

    return "\n".join(pieces)


def _looks_like_regex(pat: str) -> bool:
    return ("\\" in pat) or any(ch in pat for ch in ["[", "]", "(", ")", "|", "?", "+", "*", "{", "}"])


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def pattern_hits(text: str, pat: str) -> bool:
    """Match pat against text with a safer literal fallback."""
    if not pat:
        return False
    try:
        if _looks_like_regex(pat):
            return re.search(pat, text, flags=re.IGNORECASE) is not None
    except re.error:
        pass

    low_text = _normalize_ws(text).lower()
    low_pat = _normalize_ws(pat).lower()
    if not low_pat:
        return False

    if re.fullmatch(r"[a-z0-9][a-z0-9\s_\-/.:]*[a-z0-9]", low_pat, flags=re.IGNORECASE):
        boundary_pat = r"(?<!\w)" + re.escape(low_pat).replace(r"\ ", r"\s+") + r"(?!\w)"
        return re.search(boundary_pat, low_text, flags=re.IGNORECASE) is not None
    return low_pat in low_text


def count_pattern_hits(text: str, patterns: List[str]) -> int:
    n = 0
    for p in patterns or []:
        if isinstance(p, str) and pattern_hits(text, p):
            n += 1
    return n


def any_pattern_hits(text: str, patterns: List[str]) -> bool:
    return count_pattern_hits(text, patterns) > 0
