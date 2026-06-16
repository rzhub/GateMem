from __future__ import annotations

from typing import Any, Dict, List, Tuple

from bench.domains import detect_domain_from_episode, get_privacy_attack_types, get_safety_attack_types
from bench.eval.metrics import pattern_hits


ACTIONS = {"answer", "answer_redacted", "refuse", "no_memory"}
QUERY_TYPES = {"utility", "privacy", "safety"}
_DELETE_HINT_PATTERNS = [
    r"\bdelete\b",
    r"\bdeleted\b",
    r"\bforget\b",
    r"\bforgotten\b",
    r"\berase\b",
    r"\bremove\b",
    r"\brevok(?:e|ed|ing)\b",
    r"\bdo not retain\b",
    r"\bno longer retain\b",
]


def _non_empty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _validate_pattern_list(value: Any, *, field_name: str, required: bool = False) -> List[str]:
    errors: List[str] = []
    if value is None:
        if required:
            errors.append(f"{field_name} must be a non-empty list")
        return errors
    if not isinstance(value, list):
        errors.append(f"{field_name} must be a list")
        return errors
    if required and not value:
        errors.append(f"{field_name} must be a non-empty list")
    for idx, item in enumerate(value):
        if not _non_empty_str(item):
            errors.append(f"{field_name}[{idx}] must be a non-empty string")
    return errors


def _text_before_as_of(turns: List[Dict[str, Any]], as_of_turn_id: str) -> str:
    chunks: List[str] = []
    for t in turns:
        tid = str((t or {}).get("turn_id") or "")
        txt = str((t or {}).get("text") or "")
        if txt:
            chunks.append(txt)
        if tid == as_of_turn_id:
            break
    return "\n".join(chunks)


def _has_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(pattern_hits(text, p) for p in patterns if _non_empty_str(p))


def validate_dataset(
    *,
    episodes: List[Dict[str, Any]],
    checkpoints: List[Dict[str, Any]],
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """Validate dataset schema and selected quality heuristics."""

    errors: List[str] = []
    warnings: List[str] = []

    ep_by_id = {str(e.get("episode_id")): e for e in episodes if e.get("episode_id")}
    ep_domain = {eid: detect_domain_from_episode(ep) for eid, ep in ep_by_id.items()}

    for eid, ep in ep_by_id.items():
        turns = ep.get("turns") or []
        if not isinstance(turns, list) or not turns:
            errors.append(f"episode {eid}: missing or empty turns")
            continue
        seen = set()
        has_any_record_refs = False
        for idx, t in enumerate(turns):
            if not isinstance(t, dict):
                errors.append(f"episode {eid}: turn {idx} must be an object")
                continue
            tid = str((t or {}).get("turn_id") or "")
            if not tid:
                errors.append(f"episode {eid}: a turn is missing turn_id")
                continue
            if tid in seen:
                errors.append(f"episode {eid}: duplicate turn_id {tid}")
            seen.add(tid)
            if (t or {}).get("record_refs"):
                has_any_record_refs = True

            speaker = t.get("speaker") if isinstance(t.get("speaker"), dict) else {}
            if not _non_empty_str(speaker.get("principal_id")):
                errors.append(f"episode {eid}: turn {tid} missing non-empty speaker.principal_id")
            if not _non_empty_str(speaker.get("role")):
                errors.append(f"episode {eid}: turn {tid} missing non-empty speaker.role")
            if not _non_empty_str(t.get("text")):
                warnings.append(f"episode {eid}: turn {tid} has empty text")

        entities = (ep.get("entities") or {}) if isinstance(ep, dict) else {}
        principals = entities.get("principals") or []
        rels = entities.get("relationships") or []
        records = ep.get("records") or []

        if entities and not isinstance(entities, dict):
            errors.append(f"episode {eid}: entities must be an object when present")
        if principals and not isinstance(principals, list):
            errors.append(f"episode {eid}: entities.principals must be a list when present")
        if rels and not isinstance(rels, list):
            errors.append(f"episode {eid}: entities.relationships must be a list when present")

        if records and not has_any_record_refs:
            warnings.append(
                f"episode {eid}: has records[] but no explicit turn.record_refs; record-aware policy baselines will rely on inferred refs only."
            )

    seen_ckpt = set()
    for c in checkpoints:
        cid = str(c.get("checkpoint_id") or "")
        if not cid:
            errors.append("checkpoint missing checkpoint_id")
            continue
        if cid in seen_ckpt:
            errors.append(f"duplicate checkpoint_id: {cid}")
        seen_ckpt.add(cid)

        eid = str(c.get("episode_id") or "")
        if eid not in ep_by_id:
            errors.append(f"checkpoint {cid}: episode_id not found: {eid}")
            continue

        qtype = str(c.get("query_type") or "")
        if qtype not in QUERY_TYPES:
            errors.append(f"checkpoint {cid}: invalid query_type={qtype}")
            continue

        exp = str(c.get("expected_action") or "")
        if exp not in ACTIONS:
            errors.append(f"checkpoint {cid}: invalid expected_action={exp}")

        asker = c.get("asker") if isinstance(c.get("asker"), dict) else {}
        if not _non_empty_str(asker.get("principal_id")):
            errors.append(f"checkpoint {cid}: asker.principal_id must be a non-empty string")
        if not _non_empty_str(asker.get("role")):
            errors.append(f"checkpoint {cid}: asker.role must be a non-empty string")
        if not _non_empty_str(c.get("query_text")):
            errors.append(f"checkpoint {cid}: query_text must be a non-empty string")

        as_of = str(c.get("as_of_turn_id") or "")
        turns = ep_by_id[eid].get("turns") or []
        turn_ids = {str(t.get("turn_id")) for t in turns if isinstance(t, dict)}
        if as_of not in turn_ids:
            errors.append(f"checkpoint {cid}: as_of_turn_id={as_of} not found in episode {eid}")

        judge_spec = c.get("judge_spec")
        if not isinstance(judge_spec, dict):
            errors.append(f"checkpoint {cid}: judge_spec must be an object")
            judge_spec = {}

        js_exp = str(judge_spec.get("expected_action") or "")
        if js_exp and js_exp != exp:
            errors.append(f"checkpoint {cid}: judge_spec.expected_action ({js_exp}) != expected_action ({exp})")

        leak_targets = c.get("leak_targets")
        errors.extend(_validate_pattern_list(leak_targets, field_name=f"checkpoint {cid}: leak_targets"))
        leak_targets_list = leak_targets if isinstance(leak_targets, list) else []

        attack = c.get("attack_type")
        if qtype == "utility":
            if exp != "answer":
                errors.append(f"checkpoint {cid}: utility must have expected_action=answer")
            errors.extend(_validate_pattern_list(judge_spec.get("include"), field_name=f"checkpoint {cid}: judge_spec.include", required=True))
        elif qtype == "privacy":
            if attack is None:
                errors.append(f"checkpoint {cid}: privacy requires attack_type")
            else:
                at = str(attack)
                allowed_privacy_attacks = get_privacy_attack_types(ep_domain.get(eid, "generic"))
                if at not in allowed_privacy_attacks:
                    errors.append(f"checkpoint {cid}: privacy invalid attack_type={at} for domain={ep_domain.get(eid, 'generic')}")
            if exp not in {"refuse", "answer_redacted"}:
                errors.append(f"checkpoint {cid}: privacy expected_action must be refuse or answer_redacted")
            errors.extend(_validate_pattern_list(judge_spec.get("not_include"), field_name=f"checkpoint {cid}: judge_spec.not_include"))
            if not leak_targets_list:
                warnings.append(f"checkpoint {cid}: privacy leak_targets empty; rule-based leak checks will be weak")
        elif qtype == "safety":
            if attack is None:
                errors.append(f"checkpoint {cid}: safety requires attack_type")
            else:
                at = str(attack)
                if at not in get_safety_attack_types():
                    errors.append(f"checkpoint {cid}: safety invalid attack_type={at}")
            if exp != "no_memory":
                errors.append(f"checkpoint {cid}: safety must have expected_action=no_memory")
            if not leak_targets_list:
                errors.append(f"checkpoint {cid}: safety requires non-empty leak_targets")

        if strict and as_of in turn_ids:
            prior_text = _text_before_as_of(turns, as_of)
            include = judge_spec.get("include") if isinstance(judge_spec.get("include"), list) else []
            not_include = judge_spec.get("not_include") if isinstance(judge_spec.get("not_include"), list) else []

            if qtype == "utility" and include:
                missing = [p for p in include if _non_empty_str(p) and not pattern_hits(prior_text, p)]
                if missing:
                    warnings.append(
                        f"checkpoint {cid}: {len(missing)} utility include pattern(s) were not found before as_of_turn_id; verify support turns and temporal anchoring"
                    )
            if qtype == "privacy" and leak_targets_list and not _has_any_pattern(prior_text, leak_targets_list + not_include):
                warnings.append(
                    f"checkpoint {cid}: privacy target patterns were not found before as_of_turn_id; verify that the protected information is introduced before the attack"
                )
            if qtype == "safety" and leak_targets_list and not _has_any_pattern(prior_text, leak_targets_list):
                warnings.append(
                    f"checkpoint {cid}: safety leak_targets were not found before as_of_turn_id; verify delete/recovery chain construction"
                )
            if qtype == "safety" and not _has_any_pattern(prior_text, _DELETE_HINT_PATTERNS):
                warnings.append(
                    f"checkpoint {cid}: no obvious deletion/forgetting cue was found before as_of_turn_id; verify delete-to-attack chain"
                )

    return errors, warnings
