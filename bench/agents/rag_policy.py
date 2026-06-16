from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseMemoryAgent, Turn, Checkpoint
from bench.domains import detect_domain_from_episode
from bench.retrieval import Chunker, TfidfRetriever, EmbeddingRetriever, LangChainFaissRetriever


class PolicyRAGAgent(BaseMemoryAgent):
    """Baseline: relationship-aware policy-gated retrieval.

    Current benchmark data uses:
    - episode.entities.principals
    - episode.entities.relationships
    - turn.speaker {principal_id, role}

    and does NOT require records[] / record_refs.

    Runtime policy behavior:
    - If records[] / record_refs are available, use strict record-level gating.
    - Otherwise, use relationship- and mention-aware chunk gating based on the
      current dataset schema. This is the primary path for the current benchmark.
    """

    def __init__(
        self,
        *,
        top_k: int = 5,
        llm_mode: str = "obedient",
        llm_router: Any = None,
        query_prompt_path: str | None = None,
        retrieval_backend: str = "tfidf",  # tfidf|embedding
        chunk_mode: str = "turn",  # turn|window|chars
        chunk_window_turns: int = 5,
        chunk_max_chars: int = 4000,
        embed_router: Optional[Any] = None,
        use_faiss: bool = False,
        embedding_impl: str = "native",  # native|langchain
        embed_provider: str = "openai",  # for langchain impl
        embed_model: str = "text-embedding-3-small",  # for langchain impl
        embed_api_base: Optional[str] = None,
        embed_api_key_env: Optional[str] = None,
        embed_device: str = "cpu",
        embed_batch_size: int = 16,
        policy_filter_mode: str = "strict",  # strict|soft (only used when records exist but refs do not)
    ) -> None:
        super().__init__(
            top_k=top_k,
            llm_mode=llm_mode,
            llm_router=llm_router,
            query_prompt_path=query_prompt_path,
        )
        self.retrieval_backend = retrieval_backend
        self.embedding_impl = embedding_impl
        self._embed_router = embed_router
        self._embed_provider = embed_provider
        self._embed_model = embed_model
        self._embed_api_base = embed_api_base
        self._embed_api_key_env = embed_api_key_env
        self._embed_device = embed_device
        self._embed_batch_size = embed_batch_size
        self._use_faiss = use_faiss
        self.policy_filter_mode = policy_filter_mode

        self.chunker = Chunker(mode=chunk_mode, window_turns=chunk_window_turns, max_chars=chunk_max_chars)

        if retrieval_backend == "embedding":
            if embedding_impl == "langchain":
                self.retriever = LangChainFaissRetriever(
                    provider=embed_provider,
                    model=embed_model,
                    logger=self.logger,
                    api_base=embed_api_base,
                    api_key_env=embed_api_key_env,
                    device=embed_device,
                    batch_size=embed_batch_size,
                )
            else:
                if embed_router is None:
                    raise ValueError("retrieval_backend=embedding with embedding_impl=native requires embed_router")
                self.retriever = EmbeddingRetriever(
                    embed_router=embed_router,
                    use_faiss=use_faiss,
                    show_progress=False,
                    logger=self.logger,
                )
        else:
            self.retriever = TfidfRetriever(logger=self.logger)

        self._chunk_record_refs: Dict[str, List[str]] = {}
        self._chunk_meta: Dict[str, Dict[str, Any]] = {}
        self._embed_usage_accum: Dict[str, float] = {"input_tokens": 0.0, "total_tokens": 0.0}
        self._logged_unscoped_warning = False

    def reset(self, episode: Dict[str, Any]) -> None:
        super().reset(episode)
        self.chunker = Chunker(
            mode=self.chunker.mode,
            window_turns=self.chunker.window_turns,
            max_chars=self.chunker.max_chars,
        )
        if self.retrieval_backend == "embedding":
            if self.embedding_impl == "langchain":
                self.retriever = LangChainFaissRetriever(
                    provider=self._embed_provider,
                    model=self._embed_model,
                    logger=self.logger,
                    api_base=self._embed_api_base,
                    api_key_env=self._embed_api_key_env,
                    device=self._embed_device,
                    batch_size=self._embed_batch_size,
                )
            else:
                self.retriever = EmbeddingRetriever(
                    embed_router=self._embed_router,
                    use_faiss=self._use_faiss,
                    logger=self.logger,
                )
        else:
            self.retriever = TfidfRetriever(logger=self.logger)

        self._chunk_record_refs = {}
        self._chunk_meta = {}
        self._embed_usage_accum = {"input_tokens": 0.0, "total_tokens": 0.0}
        self._logged_unscoped_warning = False
        self._domain_key = detect_domain_from_episode(episode)
        self._build_policy_indices()

    def ingest(self, turn: Turn) -> None:
        tdict = {
            "turn_id": turn.turn_id,
            "speaker": {"principal_id": turn.speaker_principal_id, "role": turn.speaker_role},
            "text": turn.text,
            "record_refs": self._merge_record_refs(turn.record_refs, turn.text),
        }
        for ch in self.chunker.add_turn(tdict):
            self._ingest_chunk(ch)

    def _ingest_chunk(self, ch: Any) -> None:
        merged_refs = self._merge_record_refs(ch.record_refs, ch.text)
        mentions = self._infer_mentions(ch.text)
        meta = {
            "turn_ids": ch.turn_ids,
            "principal_ids": ch.principal_ids,
            "roles": ch.roles,
            "record_refs": merged_refs,
            "text": ch.text,
            "mentions": mentions,
            "content_tags": self._content_tags(ch.text),
        }
        self._chunk_record_refs[ch.chunk_id] = merged_refs
        self._chunk_meta[ch.chunk_id] = meta
        if isinstance(self.retriever, EmbeddingRetriever):
            usage = self.retriever.add(ch.chunk_id, ch.text, meta)
            it = usage.get("input_tokens")
            tt = usage.get("total_tokens")
            if isinstance(it, (int, float)):
                self._embed_usage_accum["input_tokens"] += float(it)
            if isinstance(tt, (int, float)):
                self._embed_usage_accum["total_tokens"] += float(tt)
        else:
            self.retriever.add(ch.chunk_id, ch.text, meta)  # type: ignore[arg-type]

    # ------------------------ Generic helpers ------------------------

    @staticmethod
    def _turn_num(turn_id: str) -> int:
        try:
            return int(str(turn_id).lstrip("t"))
        except Exception:
            return -1

    @staticmethod
    def _lower(val: Any) -> str:
        return str(val or "").strip().lower()

    @staticmethod
    def _first_present(obj: Dict[str, Any], keys: List[str]) -> str:
        for k in keys:
            v = obj.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    @staticmethod
    def _add_to_map_set(store: Dict[str, Set[str]], key: str, value: str) -> None:
        if key and value:
            store.setdefault(key, set()).add(value)

    @staticmethod
    def _tags_for_record(record: Dict[str, Any]) -> Set[str]:
        tags: Set[str] = set()
        for key in ("sensitivity", "record_type", "category", "subtype", "label", "kind", "scope"):
            v = record.get(key)
            if isinstance(v, list):
                for item in v:
                    s = str(item or "").strip().lower()
                    if s:
                        tags.add(s)
            else:
                s = str(v or "").strip().lower()
                if s:
                    tags.add(s)
        return tags

    @classmethod
    def _tag_contains_any(cls, tags: Set[str], needles: Set[str]) -> bool:
        return any(n in t for t in tags for n in needles)

    @staticmethod
    def _clean_name_for_aliases(name: str) -> str:
        s = re.sub(r"\([^)]*\)", "", name or "").strip().lower()
        s = re.sub(r"^(dr\.?|nurse|pharmacist|prof\.?|professor)\s+", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    @staticmethod
    def _words(s: str) -> List[str]:
        return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]

    def _principal_aliases(self, principal_id: str, display_name: str) -> List[str]:
        aliases: List[str] = []
        disp = self._clean_name_for_aliases(display_name)
        if disp:
            aliases.append(disp)
            parts = disp.split()
            if len(parts) >= 2:
                first = parts[0]
                last = parts[-1]
                if self._first_name_counts.get(first, 0) == 1 and len(first) >= 4:
                    aliases.append(first)
                if self._last_name_counts.get(last, 0) == 1 and len(last) >= 4:
                    aliases.append(last)
        pid_tail = principal_id
        for pref in [
            "patient_",
            "family_",
            "clinician_",
            "nurse_",
            "pharmacist_",
            "reception_",
            "scheduler_",
            "billing_",
            "social_worker_",
            "pm_",
            "eng_",
            "sre_",
            "security_",
            "finance_",
            "legal_",
            "sales_",
            "exec_",
            "assistant_",
            "manager_",
            "employee_",
            "student_",
            "prof_",
            "advisor_",
            "ta_",
            "ra_",
            "registrar_",
            "finaid_",
            "it_",
            "deanassist_",
            "parent_",
            "residence_",
            "residentassistant_",
            "resident_",
            "partner_",
            "adult_",
            "child_",
            "guest_",
            "cleaner_",
            "technician_",
            "manager_",
            "contact_",
            "pickup_",
            "building_",
            "caregiver_",
        ]:
            if pid_tail.startswith(pref):
                pid_tail = pid_tail[len(pref):]
                break
        pid_alias = pid_tail.replace("_", " ").strip().lower()
        if pid_alias and pid_alias != disp:
            aliases.append(pid_alias)
        # stable unique list
        out: List[str] = []
        seen: Set[str] = set()
        for a in aliases:
            a2 = a.strip().lower()
            if a2 and a2 not in seen:
                out.append(a2)
                seen.add(a2)
        return out

    @staticmethod
    def _resource_aliases(resource_id: str) -> List[str]:
        rid = str(resource_id or "").strip().lower()
        if not rid:
            return []
        aliases = [rid.replace("_", " ")]
        for pref in ["project_", "program_", "building_", "household_"]:
            if rid.startswith(pref):
                tail = rid[len(pref):].replace("_", " ")
                aliases.append(tail)
                aliases.append(pref[:-1] + " " + tail)
                break
        out: List[str] = []
        seen: Set[str] = set()
        for a in aliases:
            a2 = re.sub(r"\s+", " ", a).strip().lower()
            if a2 and a2 not in seen:
                out.append(a2)
                seen.add(a2)
        return out

    @staticmethod
    def _compile_alias_patterns(aliases: List[str]) -> List[re.Pattern]:
        patterns: List[re.Pattern] = []
        seen: Set[str] = set()
        for alias in sorted(aliases, key=len, reverse=True):
            a = alias.strip().lower()
            if not a or a in seen:
                continue
            seen.add(a)
            try:
                patterns.append(re.compile(rf"(?<!\w){re.escape(a)}(?!\w)", flags=re.IGNORECASE))
            except re.error:
                continue
        return patterns

    def _content_tags(self, text: str) -> Set[str]:
        tl = (text or "").lower()
        tags: Set[str] = set()
        groups = {
            "schedule": [
                "schedule", "scheduled", "calendar", "date", "time", "window", "slot", "appointment",
                "arrival", "pickup", "dropoff", "follow-up", "follow up", "hearing", "showcase", "dry-run",
                "dry run", "meeting", "location", "where", "when", "through friday", "through monday",
            ],
            "logistics": [
                "entrance", "door", "lobby", "check in", "check-in", "wait outside", "call on arrival",
                "approved rooms", "presence rule", "transport", "release", "logistics", "broad wording",
                "safe broad", "safe wording", "summary", "status line",
            ],
            "credential": [
                "token", "pin", "code", "credential", "password", "passphrase", "secret", "alarm", "badge",
                "camera-sharing", "camera sharing", "intercom", "access scope", "active token",
            ],
            "funding": ["budget", "discount", "stipend", "grant", "aid", "funding", "scholarship", "amount"],
            "legal": ["contract", "vendor", "privileged", "legal", "terms", "msa", "sow"],
            "security": ["incident", "diagnosis", "security", "coordination", "staging token", "access scope"],
            "health": [
                "medication", "medications", "pain", "nausea", "symptom", "symptoms", "emergency", "scan",
                "imaging", "dose", "callback", "cardioversion", "hypertension", "gdm", "mavyret", "afib",
            ],
            "billing": ["billing", "insurance", "coverage", "copay", "claim"],
            "support": ["support", "social work", "counsel", "check in", "check-in", "case"],
            "registration": ["registration", "registrar", "practicum", "title", "hold", "open"],
            "housing": ["room", "housing", "hall", "dorm", "beacon hall", "commons", "reassignment", "building"],
            "pickup": ["pickup", "school", "robotics", "release"],
            "care": ["care", "caregiver", "approved rooms", "care window", "medication pass"],
            "cleaning": ["cleaning", "cleaner"],
            "repair": ["repair", "technician", "appliance"],
            "device": ["camera", "device", "setup", "local setup", "sharing scope"],
        }
        for tag, needles in groups.items():
            if any(n in tl for n in needles):
                tags.add(tag)
        return tags

    def _infer_mentions(self, text: str) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {
            "principal_ids": set(),
            "patient_ids": set(),
            "student_ids": set(),
            "resident_ids": set(),
            "child_ids": set(),
            "guest_ids": set(),
            "project_ids": set(),
            "program_ids": set(),
            "building_ids": set(),
            "household_ids": set(),
        }
        if not text:
            return out
        for pid, patterns in self._principal_patterns.items():
            if any(p.search(text) for p in patterns):
                out["principal_ids"].add(pid)
                role = self._principal_role_by_id.get(pid, "")
                if role == "patient":
                    out["patient_ids"].add(pid)
                elif role == "student":
                    out["student_ids"].add(pid)
                elif role in {"primary_resident", "partner_spouse", "adult_child", "minor_child"}:
                    out["resident_ids"].add(pid)
                if role == "minor_child":
                    out["child_ids"].add(pid)
                if role == "guest":
                    out["guest_ids"].add(pid)
        for rid, patterns in self._resource_patterns.items():
            if any(p.search(text) for p in patterns):
                self._add_mentioned_resource(out, rid)
        return out

    def _add_mentioned_resource(self, mentions: Dict[str, Set[str]], rid: str) -> None:
        if rid.startswith("project_"):
            mentions["project_ids"].add(rid)
        elif rid.startswith("program_"):
            mentions["program_ids"].add(rid)
        elif rid.startswith("building_"):
            mentions["building_ids"].add(rid)
        elif rid.endswith("_house") or rid.startswith("household_"):
            mentions["household_ids"].add(rid)

    def _merge_mentions_with_speakers(self, meta: Dict[str, Any]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {
            k: set(v) for k, v in (meta.get("mentions") or {}).items()
        }
        for k in ["principal_ids", "patient_ids", "student_ids", "resident_ids", "child_ids", "guest_ids", "project_ids", "program_ids", "building_ids", "household_ids"]:
            out.setdefault(k, set())
        for pid in (meta.get("principal_ids") or []):
            if pid:
                out["principal_ids"].add(str(pid))
                role = self._principal_role_by_id.get(str(pid), "")
                if role == "patient":
                    out["patient_ids"].add(str(pid))
                elif role == "student":
                    out["student_ids"].add(str(pid))
                elif role in {"primary_resident", "partner_spouse", "adult_child", "elder_family_member", "minor_child"}:
                    out["resident_ids"].add(str(pid))
                if role == "minor_child":
                    out["child_ids"].add(str(pid))
                if role == "guest":
                    out["guest_ids"].add(str(pid))
                for proj in self._projects_by_principal.get(str(pid), set()):
                    out["project_ids"].add(proj)
                for prog in self._programs_by_principal.get(str(pid), set()):
                    out["program_ids"].add(prog)
                for bid in self._buildings_by_principal.get(str(pid), set()):
                    out["building_ids"].add(bid)
                for hid in self._households_by_principal.get(str(pid), set()):
                    out["household_ids"].add(hid)
                for patient_id in self._patients_by_principal.get(str(pid), set()):
                    out["patient_ids"].add(patient_id)
                for student_id in self._students_by_principal.get(str(pid), set()):
                    out["student_ids"].add(student_id)
                for child_id in self._children_by_principal.get(str(pid), set()):
                    out["child_ids"].add(child_id)
        return out

    @staticmethod
    def _scope_allowed_tags(scope: str) -> Optional[Set[str]]:
        scope_l = str(scope or "").strip().lower()
        if not scope_l:
            return None
        allowed: Set[str] = set()
        if any(tok in scope_l for tok in {"schedule", "scheduled", "visit", "calendar", "same_day", "wednesday", "monday", "tuesday"}):
            allowed |= {"schedule"}
        if any(tok in scope_l for tok in {"logistics", "wait_outside", "call_only", "call", "lobby"}):
            allowed |= {"logistics"}
        if any(tok in scope_l for tok in {"pickup", "school", "robotics", "release"}):
            allowed |= {"pickup", "schedule", "logistics"}
        if any(tok in scope_l for tok in {"credential", "token", "pin", "code", "badge", "intercom"}):
            allowed |= {"credential"}
        if any(tok in scope_l for tok in {"local_setup", "setup", "device", "camera"}):
            allowed |= {"device"}
        if any(tok in scope_l for tok in {"cleaning", "cleaner"}):
            allowed |= {"cleaning", "schedule", "logistics"}
        if any(tok in scope_l for tok in {"repair", "appliance", "technician"}):
            allowed |= {"repair", "schedule", "logistics"}
        if any(tok in scope_l for tok in {"care", "approved_rooms", "care_window", "medication pass"}):
            allowed |= {"care", "schedule", "logistics"}
        if any(tok in scope_l for tok in {"building", "intercom", "lobby"}):
            allowed |= {"housing", "schedule", "logistics"}
        return allowed or None

    def _scope_allows_tags(self, scope: str, tags: Set[str]) -> bool:
        allowed = self._scope_allowed_tags(scope)
        if allowed is None:
            return True
        scoped_tags = set(tags or set()) & {
            "schedule", "logistics", "pickup", "credential", "device", "cleaning", "repair", "care", "housing"
        }
        if not scoped_tags:
            return True
        return scoped_tags.issubset(allowed)

    # ------------------------ Index construction ------------------------

    def _build_policy_indices(self) -> None:
        self._record_index: Dict[str, Dict[str, Any]] = {}

        self._principal_role_by_id: Dict[str, str] = {}
        self._principal_display_by_id: Dict[str, str] = {}
        self._principal_patterns: Dict[str, List[re.Pattern]] = {}
        self._resource_patterns: Dict[str, List[re.Pattern]] = {}

        self._projects_by_principal: Dict[str, Set[str]] = {}
        self._project_members_by_project: Dict[str, Set[str]] = {}
        self._project_budget_owners_by_project: Dict[str, Set[str]] = {}
        self._project_legal_reviewers_by_project: Dict[str, Set[str]] = {}
        self._project_contractors_by_project: Dict[str, Set[str]] = {}
        self._project_executive_sponsors_by_project: Dict[str, Set[str]] = {}
        self._contractor_projects: Dict[str, Set[str]] = {}
        self._assistant_delegate_of: Dict[str, Set[str]] = {}
        self._managed_employees_by_manager: Dict[str, Set[str]] = {}

        self._programs_by_principal: Dict[str, Set[str]] = {}
        self._program_members_by_program: Dict[str, Set[str]] = {}
        self._buildings_by_principal: Dict[str, Set[str]] = {}
        self._students_by_principal: Dict[str, Set[str]] = {}
        self._students_by_staff_any: Dict[str, Set[str]] = {}
        self._staff_by_student_any: Dict[str, Set[str]] = {}
        self._guardian_students_by_guardian: Dict[str, Set[str]] = {}
        self._delegate_for_staff: Dict[str, Set[str]] = {}

        self._households_by_principal: Dict[str, Set[str]] = {}
        self._household_members_by_household: Dict[str, Set[str]] = {}
        self._guests_by_host: Dict[str, Set[str]] = {}
        self._children_by_principal: Dict[str, Set[str]] = {}
        self._service_scope_by_principal: Dict[str, str] = {}
        self._pickup_children_by_principal: Dict[str, Set[str]] = {}
        self._cared_for_by_principal: Dict[str, Set[str]] = {}

        self._patients_by_principal: Dict[str, Set[str]] = {}
        self._assigned_clinicians_by_patient: Dict[str, Set[str]] = {}
        self._assigned_patients_by_clinician: Dict[str, Set[str]] = {}
        self._family_patients_by_family: Dict[str, Set[str]] = {}
        self._family_members_by_patient: Dict[str, Set[str]] = {}
        self._family_scope_by_family: Dict[str, str] = {}

        if not self.episode:
            return

        principals = ((self.episode.get("entities") or {}).get("principals") or [])
        first_names: Dict[str, int] = {}
        last_names: Dict[str, int] = {}
        for p in principals:
            pid = self._first_present(p, ["principal_id", "id"])
            role = self._first_present(p, ["role"])
            display = self._first_present(p, ["display_name", "name", "label"])
            if pid:
                self._principal_role_by_id[pid] = role
                self._principal_display_by_id[pid] = display
                cleaned = self._clean_name_for_aliases(display)
                parts = cleaned.split()
                if parts:
                    first_names[parts[0]] = first_names.get(parts[0], 0) + 1
                    last_names[parts[-1]] = last_names.get(parts[-1], 0) + 1
        self._first_name_counts = first_names
        self._last_name_counts = last_names
        for pid, display in self._principal_display_by_id.items():
            self._principal_patterns[pid] = self._compile_alias_patterns(self._principal_aliases(pid, display))

        for r in (self.episode.get("records") or []):
            rid = r.get("record_id")
            if rid:
                self._record_index[str(rid)] = r

        rels = ((self.episode.get("entities") or {}).get("relationships") or [])
        resource_ids: Set[str] = set()
        for rel in rels:
            rtype = self._lower(rel.get("type"))
            if not rtype:
                continue
            # collect resource ids for matcher compilation
            for key in ["project_id", "program_id", "building_id", "household_id"]:
                rv = self._first_present(rel, [key])
                if rv:
                    resource_ids.add(rv)

            if rtype in {"assigned_clinician", "covering_clinician"}:
                pid = self._first_present(rel, ["patient_id", "for_principal_id"])
                cid = self._first_present(rel, ["clinician_id", "principal_id"])
                self._add_to_map_set(self._assigned_clinicians_by_patient, pid, cid)
                self._add_to_map_set(self._assigned_patients_by_clinician, cid, pid)
                self._add_to_map_set(self._patients_by_principal, cid, pid)
                continue

            if rtype == "family_member":
                fid = self._first_present(rel, ["family_id", "principal_id"])
                pid = self._first_present(rel, ["patient_id", "for_principal_id"])
                self._add_to_map_set(self._family_patients_by_family, fid, pid)
                self._add_to_map_set(self._family_members_by_patient, pid, fid)
                self._family_scope_by_family[fid] = self._lower(rel.get("access_level"))
                self._add_to_map_set(self._patients_by_principal, fid, pid)
                continue

            if rtype in {"project_member", "research_project_member"}:
                project_id = self._first_present(rel, ["project_id"])
                principal_id = self._first_present(rel, ["principal_id"])
                self._add_to_map_set(self._project_members_by_project, project_id, principal_id)
                self._add_to_map_set(self._projects_by_principal, principal_id, project_id)
                continue

            if rtype == "budget_owner":
                project_id = self._first_present(rel, ["project_id"])
                principal_id = self._first_present(rel, ["principal_id"])
                self._add_to_map_set(self._project_budget_owners_by_project, project_id, principal_id)
                self._add_to_map_set(self._projects_by_principal, principal_id, project_id)
                continue

            if rtype == "legal_reviewer":
                project_id = self._first_present(rel, ["project_id"])
                principal_id = self._first_present(rel, ["principal_id"])
                self._add_to_map_set(self._project_legal_reviewers_by_project, project_id, principal_id)
                self._add_to_map_set(self._projects_by_principal, principal_id, project_id)
                continue

            if rtype == "executive_sponsor":
                project_id = self._first_present(rel, ["project_id"])
                principal_id = self._first_present(rel, ["principal_id"])
                self._add_to_map_set(self._project_executive_sponsors_by_project, project_id, principal_id)
                self._add_to_map_set(self._projects_by_principal, principal_id, project_id)
                continue

            if rtype == "contractor_for_project":
                contractor_id = self._first_present(rel, ["principal_id", "contractor_id"])
                project_id = self._first_present(rel, ["project_id"])
                self._add_to_map_set(self._contractor_projects, contractor_id, project_id)
                self._add_to_map_set(self._project_contractors_by_project, project_id, contractor_id)
                self._add_to_map_set(self._projects_by_principal, contractor_id, project_id)
                continue

            if rtype == "executive_delegate":
                assistant_id = self._first_present(rel, ["principal_id", "assistant_id"])
                executive_id = self._first_present(rel, ["for_principal_id", "executive_id"])
                if assistant_id and executive_id and self._lower(rel.get("access_scope")) in {"scheduling_only", "schedule_only", "calendar_only"}:
                    self._add_to_map_set(self._assistant_delegate_of, assistant_id, executive_id)
                continue

            if rtype == "people_manager":
                manager_id = self._first_present(rel, ["principal_id"])
                employee_id = self._first_present(rel, ["manages", "for_principal_id", "employee_id"])
                self._add_to_map_set(self._managed_employees_by_manager, manager_id, employee_id)
                continue

            if rtype in {"program_member", "faculty_lead"}:
                program_id = self._first_present(rel, ["program_id"])
                principal_id = self._first_present(rel, ["principal_id"])
                self._add_to_map_set(self._program_members_by_program, program_id, principal_id)
                self._add_to_map_set(self._programs_by_principal, principal_id, program_id)
                continue

            if rtype in {"advisor_for", "research_supervisor", "funding_case_owner", "registrar_case_owner", "residence_case_owner"}:
                staff_id = self._first_present(rel, ["principal_id"])
                student_id = self._first_present(rel, ["for_principal_id", "student_id"])
                self._add_to_map_set(self._students_by_staff_any, staff_id, student_id)
                self._add_to_map_set(self._staff_by_student_any, student_id, staff_id)
                self._add_to_map_set(self._students_by_principal, staff_id, student_id)
                if rtype == "residence_case_owner":
                    self._add_to_map_set(self._buildings_by_principal, staff_id, self._first_present(rel, ["building_id"]))
                continue

            if rtype in {"it_support", "building_access_support", "resident_assistant_for"}:
                pid = self._first_present(rel, ["principal_id"])
                project_id = self._first_present(rel, ["project_id"])
                building_id = self._first_present(rel, ["building_id"])
                self._add_to_map_set(self._projects_by_principal, pid, project_id)
                self._add_to_map_set(self._buildings_by_principal, pid, building_id)
                continue

            if rtype == "delegate":
                assistant_id = self._first_present(rel, ["principal_id"])
                staff_id = self._first_present(rel, ["for_principal_id"])
                if assistant_id and staff_id and self._lower(rel.get("access_scope")) in {"scheduling_only", "schedule_only"}:
                    self._add_to_map_set(self._delegate_for_staff, assistant_id, staff_id)
                continue

            if rtype == "parent_of":
                parent_id = self._first_present(rel, ["principal_id"])
                student_id = self._first_present(rel, ["for_principal_id", "student_id"])
                self._add_to_map_set(self._guardian_students_by_guardian, parent_id, student_id)
                self._add_to_map_set(self._students_by_principal, parent_id, student_id)
                continue

            if rtype == "household_member":
                hid = self._first_present(rel, ["household_id"])
                pid = self._first_present(rel, ["principal_id"])
                self._add_to_map_set(self._household_members_by_household, hid, pid)
                self._add_to_map_set(self._households_by_principal, pid, hid)
                continue

            if rtype == "host_of_guest":
                pid = self._first_present(rel, ["principal_id"])
                gid = self._first_present(rel, ["guest_id"])
                hid = self._first_present(rel, ["household_id"])
                self._add_to_map_set(self._guests_by_host, pid, gid)
                self._add_to_map_set(self._children_by_principal, pid, gid)  # generic dependent-like visibility
                self._add_to_map_set(self._households_by_principal, pid, hid)
                self._add_to_map_set(self._households_by_principal, gid, hid)
                continue

            if rtype in {"service_provider", "logistics_delegate", "trusted_backup", "device_setup_delegate"}:
                pid = self._first_present(rel, ["principal_id"])
                hid = self._first_present(rel, ["household_id"])
                self._add_to_map_set(self._households_by_principal, pid, hid)
                if pid:
                    self._service_scope_by_principal[pid] = self._lower(rel.get("access_scope"))
                continue

            if rtype == "pickup_authorized":
                pid = self._first_present(rel, ["principal_id"])
                child_id = self._first_present(rel, ["for_principal_id", "child_id"])
                self._add_to_map_set(self._pickup_children_by_principal, pid, child_id)
                self._add_to_map_set(self._children_by_principal, pid, child_id)
                if pid:
                    self._service_scope_by_principal[pid] = self._lower(rel.get("access_scope"))
                continue

            if rtype == "caregiver_for":
                pid = self._first_present(rel, ["principal_id"])
                target_id = self._first_present(rel, ["for_principal_id"])
                self._add_to_map_set(self._cared_for_by_principal, pid, target_id)
                self._add_to_map_set(self._children_by_principal, pid, target_id)
                if pid:
                    self._service_scope_by_principal[pid] = self._lower(rel.get("access_scope"))
                continue

        for rid in resource_ids:
            self._resource_patterns[rid] = self._compile_alias_patterns(self._resource_aliases(rid))

    # ------------------------ Record-aware gating (future-compatible) ------------------------

    def _is_deleted_as_of(self, record: Dict[str, Any], as_of_turn_id: str) -> bool:
        del_tid = record.get("deleted_at_turn")
        if not del_tid:
            return False
        return self._turn_num(str(del_tid)) <= self._turn_num(as_of_turn_id)

    def _record_owner(self, record: Dict[str, Any]) -> str:
        return self._first_present(
            record,
            ["owner_principal", "owner_principal_id", "owner_id", "patient_id", "student_id", "employee_id", "resident_id"],
        )

    def _requester_can_access_record(self, checkpoint: Checkpoint, record: Dict[str, Any]) -> bool:
        requester_id = checkpoint.asker_principal_id
        requester_role = checkpoint.asker_role
        owner = self._record_owner(record)
        sensitivity = self._lower(record.get("sensitivity"))
        tags = self._tags_for_record(record)

        if requester_id == owner:
            return True
        allowed_principals = {str(x) for x in (record.get("allowed_principal_ids") or []) if str(x)}
        if requester_id in allowed_principals:
            return True
        allowed_roles = {str(x) for x in (record.get("allowed_roles") or []) if str(x)}
        if requester_role in allowed_roles:
            return True
        scoped_principals = {str(x) for x in (record.get("assigned_principal_ids") or record.get("scope_principal_ids") or []) if str(x)}
        if requester_id in scoped_principals:
            return True

        domain = getattr(self, "_domain_key", "generic")
        if domain == "medical":
            if requester_role == "family_member" and owner in self._family_patients_by_family.get(requester_id, set()):
                return "schedule" in tags or sensitivity in {"summary_only", "health_low", "health_summary"}
            if requester_role in {"clinician", "nurse", "pharmacist"} and owner in self._assigned_patients_by_clinician.get(requester_id, set()):
                return True
            return False

        if domain == "household":
            if requester_id == owner:
                return True
        return False

    # ------------------------ Relationship-aware gating (primary path) ------------------------

    def _role_allows_tags(self, requester_role: str, tags: Set[str], requester_id: str = "") -> bool:
        role = self._lower(requester_role)
        full_access_roles = {
            "patient", "clinician", "product_manager", "engineer", "employee", "manager", "sre", "security",
            "legal", "finance", "executive", "primary_resident", "partner_spouse", "adult_child",
            "elder_family_member", "student", "professor", "advisor", "ta", "ra",
        }
        if role in full_access_roles:
            return True
        if role == "family_member":
            return bool(tags & {"schedule", "logistics"})
        if role in {"reception", "scheduler"}:
            return bool(tags & {"schedule", "logistics"})
        if role == "billing":
            return bool(tags & {"billing", "schedule", "logistics"})
        if role in {"social_worker", "counselor"}:
            return bool(tags & {"support", "schedule", "logistics"})
        if role in {"nurse", "lab_tech"}:
            return bool(tags & {"health", "schedule", "logistics"})
        if role == "pharmacist":
            return bool(tags & {"health", "credential"})
        if role == "executive_assistant":
            return bool(tags & {"schedule", "logistics"})
        if role == "contractor":
            return not bool(tags & {"funding", "legal", "credential", "security"})
        if role == "sales":
            return not bool(tags & {"credential", "security", "legal"})
        if role == "hr":
            return bool(tags & {"schedule", "logistics"})
        if role in {"campus_it", "home_it_helper"}:
            return bool(tags & {"credential", "device", "housing"})
        if role == "financial_aid":
            return bool(tags & {"funding", "schedule", "logistics"})
        if role == "registrar":
            return bool(tags & {"registration", "schedule", "logistics"})
        if role in {"department_admin", "dean_assistant"}:
            return bool(tags & {"schedule", "logistics", "registration"})
        if role in {"residence_staff", "resident_assistant"}:
            return bool(tags & {"housing", "schedule", "logistics"})
        if role == "parent_guardian":
            return bool(tags & {"schedule", "logistics"})
        if role == "household_manager":
            return bool(tags & {"schedule", "logistics", "cleaning", "repair"})
        if role == "trusted_contact":
            return bool(tags & {"schedule", "logistics"})
        if role == "driver_pickup_contact":
            return bool(tags & {"pickup", "schedule", "logistics"})
        if role == "caregiver":
            return bool(tags & {"care", "schedule", "logistics"})
        if role == "cleaner":
            return bool(tags & {"cleaning", "schedule", "logistics"})
        if role == "technician":
            return bool(tags & {"repair", "schedule", "logistics"})
        if role == "building_staff":
            return bool(tags & {"credential", "device", "schedule", "logistics"})
        if role == "guest":
            return bool(tags & {"schedule", "logistics"})
        if role == "minor_child":
            return bool(tags & {"pickup", "schedule", "logistics"})
        return False

    def _medical_allowed_patients(self, checkpoint: Checkpoint, query_mentions: Dict[str, Set[str]]) -> Set[str]:
        rid = checkpoint.asker_principal_id
        role = self._lower(checkpoint.asker_role)
        if role == "patient":
            return {rid}
        if role == "family_member":
            return set(self._family_patients_by_family.get(rid, set()))
        if role == "clinician":
            return set(self._assigned_patients_by_clinician.get(rid, set()))
        if role in {"nurse", "pharmacist", "reception", "scheduler", "billing", "social_worker", "lab_tech"}:
            return set(query_mentions.get("patient_ids", set()))
        return set(query_mentions.get("patient_ids", set()))

    def _medical_visible_principals(self, allowed_patients: Set[str]) -> Set[str]:
        visible: Set[str] = set(allowed_patients)
        generic_support = {pid for pid, role in self._principal_role_by_id.items() if role in {"reception", "scheduler", "billing", "social_worker", "lab_tech", "nurse", "pharmacist", "agent"}}
        visible |= generic_support
        for patient_id in allowed_patients:
            visible |= self._assigned_clinicians_by_patient.get(patient_id, set())
            visible |= self._family_members_by_patient.get(patient_id, set())
        return visible

    def _medical_chunk_access(self, checkpoint: Checkpoint, meta: Dict[str, Any], query_mentions: Dict[str, Set[str]]) -> bool:
        allowed_patients = self._medical_allowed_patients(checkpoint, query_mentions)
        if not allowed_patients:
            return False
        tags = set(meta.get("content_tags") or set())
        if not self._role_allows_tags(checkpoint.asker_role, tags, checkpoint.asker_principal_id):
            return False
        requester_role = self._lower(checkpoint.asker_role)
        if requester_role == "family_member":
            if not self._scope_allows_tags(self._family_scope_by_family.get(checkpoint.asker_principal_id, ""), tags):
                return False
        mentions = self._merge_mentions_with_speakers(meta)
        chunk_patients = set(mentions.get("patient_ids", set()))
        for pid in mentions.get("principal_ids", set()):
            chunk_patients |= self._patients_by_principal.get(pid, set())
        if chunk_patients and not chunk_patients.issubset(allowed_patients):
            return False
        if chunk_patients:
            return True
        return set(meta.get("principal_ids") or []).issubset(self._medical_visible_principals(allowed_patients))

    def _office_allowed_projects(self, requester_id: str) -> Set[str]:
        projects = set(self._projects_by_principal.get(requester_id, set())) | set(self._contractor_projects.get(requester_id, set()))
        for exec_id in self._assistant_delegate_of.get(requester_id, set()):
            projects |= self._projects_by_principal.get(exec_id, set())
        return projects

    def _office_visible_principals(self, allowed_projects: Set[str], requester_id: str) -> Set[str]:
        visible: Set[str] = {requester_id}
        for project_id in allowed_projects:
            visible |= self._project_members_by_project.get(project_id, set())
            visible |= self._project_budget_owners_by_project.get(project_id, set())
            visible |= self._project_legal_reviewers_by_project.get(project_id, set())
            visible |= self._project_contractors_by_project.get(project_id, set())
            visible |= self._project_executive_sponsors_by_project.get(project_id, set())
        visible |= self._managed_employees_by_manager.get(requester_id, set())
        return visible

    def _office_chunk_access(self, checkpoint: Checkpoint, meta: Dict[str, Any], query_mentions: Dict[str, Set[str]]) -> bool:
        requester_id = checkpoint.asker_principal_id
        role = self._lower(checkpoint.asker_role)
        tags = set(meta.get("content_tags") or set())
        if not self._role_allows_tags(role, tags, requester_id):
            return False
        allowed_projects = self._office_allowed_projects(requester_id)
        if role in {"manager", "hr"}:
            allowed_projects |= set(query_mentions.get("project_ids", set()))
        mentions = self._merge_mentions_with_speakers(meta)
        chunk_projects = set(mentions.get("project_ids", set()))
        if not chunk_projects:
            for pid in mentions.get("principal_ids", set()):
                chunk_projects |= self._projects_by_principal.get(pid, set())
        if chunk_projects and allowed_projects:
            if not chunk_projects.issubset(allowed_projects):
                return False
            return True
        return set(meta.get("principal_ids") or []).issubset(self._office_visible_principals(allowed_projects, requester_id))

    def _education_allowed(self, checkpoint: Checkpoint, query_mentions: Dict[str, Set[str]]) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        rid = checkpoint.asker_principal_id
        role = self._lower(checkpoint.asker_role)
        students: Set[str] = set()
        projects: Set[str] = set()
        programs: Set[str] = set()
        buildings: Set[str] = set()

        if role == "student":
            students.add(rid)
        if role in {"professor", "advisor", "ta", "ra", "counselor", "financial_aid", "registrar", "residence_staff", "resident_assistant", "campus_it"}:
            students |= self._students_by_staff_any.get(rid, set())
            projects |= self._projects_by_principal.get(rid, set())
            programs |= self._programs_by_principal.get(rid, set())
            buildings |= self._buildings_by_principal.get(rid, set())
        if role in {"dean_assistant"}:
            for staff_id in self._delegate_for_staff.get(rid, set()):
                students |= self._students_by_staff_any.get(staff_id, set())
                projects |= self._projects_by_principal.get(staff_id, set())
                programs |= self._programs_by_principal.get(staff_id, set())
                buildings |= self._buildings_by_principal.get(staff_id, set())
        if role == "parent_guardian":
            students |= self._guardian_students_by_guardian.get(rid, set())
        if role == "department_admin":
            students |= set(query_mentions.get("student_ids", set()))
            projects |= set(query_mentions.get("project_ids", set()))
            programs |= set(query_mentions.get("program_ids", set()))
            buildings |= set(query_mentions.get("building_ids", set()))
        return students, projects, programs, buildings

    def _education_visible_principals(self, students: Set[str], projects: Set[str], programs: Set[str], buildings: Set[str], requester_id: str) -> Set[str]:
        visible: Set[str] = {requester_id}
        for student_id in students:
            visible |= self._staff_by_student_any.get(student_id, set())
            visible.add(student_id)
        for project_id in projects:
            visible |= self._project_members_by_project.get(project_id, set())
        for program_id in programs:
            visible |= self._program_members_by_program.get(program_id, set())
        for bid in buildings:
            visible |= {pid for pid, bs in self._buildings_by_principal.items() if bid in bs}
        return visible

    def _education_chunk_access(self, checkpoint: Checkpoint, meta: Dict[str, Any], query_mentions: Dict[str, Set[str]]) -> bool:
        role = self._lower(checkpoint.asker_role)
        tags = set(meta.get("content_tags") or set())
        if not self._role_allows_tags(role, tags, checkpoint.asker_principal_id):
            return False
        students, projects, programs, buildings = self._education_allowed(checkpoint, query_mentions)
        if not (students or projects or programs or buildings):
            return False
        mentions = self._merge_mentions_with_speakers(meta)
        chunk_students = set(mentions.get("student_ids", set()))
        chunk_projects = set(mentions.get("project_ids", set()))
        chunk_programs = set(mentions.get("program_ids", set()))
        chunk_buildings = set(mentions.get("building_ids", set()))
        if chunk_students and not chunk_students.issubset(students):
            return False
        if chunk_projects and projects and not chunk_projects.issubset(projects):
            return False
        if chunk_programs and programs and not chunk_programs.issubset(programs):
            return False
        if chunk_buildings and buildings and not chunk_buildings.issubset(buildings):
            return False
        if chunk_students or chunk_projects or chunk_programs or chunk_buildings:
            return True
        visible = self._education_visible_principals(students, projects, programs, buildings, checkpoint.asker_principal_id)
        return set(meta.get("principal_ids") or []).issubset(visible)

    def _household_allowed(self, checkpoint: Checkpoint, query_mentions: Dict[str, Set[str]]) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        rid = checkpoint.asker_principal_id
        role = self._lower(checkpoint.asker_role)
        households = set(self._households_by_principal.get(rid, set()))
        children = set(self._pickup_children_by_principal.get(rid, set()))
        guests = set(self._guests_by_host.get(rid, set()))
        cared = set(self._cared_for_by_principal.get(rid, set()))
        if role == "guest":
            guests.add(rid)
        if role in {"primary_resident", "partner_spouse", "adult_child", "elder_family_member", "minor_child"}:
            children |= self._children_by_principal.get(rid, set())
            guests |= self._guests_by_host.get(rid, set())
        if role in {"household_manager", "trusted_contact", "home_it_helper", "building_staff", "cleaner", "technician", "driver_pickup_contact", "caregiver"} and not households:
            households |= set(query_mentions.get("household_ids", set()))
        return households, children, guests, cared

    def _household_visible_principals(self, households: Set[str], requester_id: str) -> Set[str]:
        visible: Set[str] = {requester_id}
        for hid in households:
            visible |= self._household_members_by_household.get(hid, set())
            visible |= {pid for pid, hs in self._households_by_principal.items() if hid in hs}
        return visible

    def _household_chunk_access(self, checkpoint: Checkpoint, meta: Dict[str, Any], query_mentions: Dict[str, Set[str]]) -> bool:
        role = self._lower(checkpoint.asker_role)
        tags = set(meta.get("content_tags") or set())
        if not self._role_allows_tags(role, tags, checkpoint.asker_principal_id):
            return False
        service_scope = self._service_scope_by_principal.get(checkpoint.asker_principal_id, "")
        if service_scope and not self._scope_allows_tags(service_scope, tags):
            return False
        households, children, guests, cared = self._household_allowed(checkpoint, query_mentions)
        mentions = self._merge_mentions_with_speakers(meta)
        chunk_households = set(mentions.get("household_ids", set()))
        chunk_children = set(mentions.get("child_ids", set()))
        chunk_guests = set(mentions.get("guest_ids", set()))
        if chunk_households and households and not chunk_households.issubset(households):
            return False
        if chunk_children and children and not chunk_children.issubset(children):
            return False
        if chunk_guests and guests and not chunk_guests.issubset(guests):
            return False
        if cared:
            mentioned = set(mentions.get("principal_ids", set()))
            if mentioned & cared:
                return True
        if chunk_households or chunk_children or chunk_guests:
            return True
        return set(meta.get("principal_ids") or []).issubset(self._household_visible_principals(households, checkpoint.asker_principal_id))

    def _requester_can_access_chunk_without_records(self, checkpoint: Checkpoint, chunk_id: str) -> bool:
        meta = self._chunk_meta.get(chunk_id) or {}
        if not meta:
            return False
        query_mentions = self._infer_mentions(checkpoint.query_text)
        domain = getattr(self, "_domain_key", "generic")
        if domain == "medical":
            return self._medical_chunk_access(checkpoint, meta, query_mentions)
        if domain == "office":
            return self._office_chunk_access(checkpoint, meta, query_mentions)
        if domain == "education":
            return self._education_chunk_access(checkpoint, meta, query_mentions)
        if domain == "household":
            return self._household_chunk_access(checkpoint, meta, query_mentions)
        # generic conservative fallback: only same-speaker or explicitly mentioned requester
        mentions = self._merge_mentions_with_speakers(meta)
        return checkpoint.asker_principal_id in mentions.get("principal_ids", set())

    def _allowed_chunk_ids(self, checkpoint: Checkpoint) -> Set[str]:
        allowed_ids: Set[str] = set()
        as_of = checkpoint.as_of_turn_id
        have_records = bool(self._record_index)

        for chunk_id, refs in self._chunk_record_refs.items():
            if refs:
                ok = True
                for rid in refs:
                    rec = self._record_index.get(rid)
                    if not rec:
                        ok = False
                        break
                    if self._is_deleted_as_of(rec, as_of):
                        ok = False
                        break
                    if not self._requester_can_access_record(checkpoint, rec):
                        ok = False
                        break
                if ok:
                    allowed_ids.add(chunk_id)
                continue

            if have_records:
                if self.policy_filter_mode == "soft":
                    allowed_ids.add(chunk_id)
                else:
                    if not self._logged_unscoped_warning:
                        self.logger.warning(
                            "rag_policy: some chunks have no record_refs while records[] are present; strict mode blocks those unscoped chunks."
                        )
                        self._logged_unscoped_warning = True
                continue

            if self._requester_can_access_chunk_without_records(checkpoint, chunk_id):
                allowed_ids.add(chunk_id)

        return allowed_ids

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        for ch in self.chunker.finalize():
            self._ingest_chunk(ch)

        allow_ids = self._allowed_chunk_ids(checkpoint)
        self.logger.info(
            "Policy gate: allowed_chunks=%d (mode=%s, domain=%s)",
            len(allow_ids),
            self.policy_filter_mode,
            getattr(self, "_domain_key", "generic"),
        )

        if isinstance(self.retriever, EmbeddingRetriever):
            hits, retrieval_s, q_usage = self.retriever.search(
                checkpoint.query_text, top_k=self.top_k, allow_chunk_ids=allow_ids
            )
        else:
            hits, retrieval_s = self.retriever.search(
                checkpoint.query_text, top_k=self.top_k, allow_chunk_ids=allow_ids
            )
            q_usage = {}

        self.logger.info(
            "Retrieval: backend=%s docs=%d top_k=%d hits=%d (t=%0.3fs)",
            self.retrieval_backend,
            getattr(self.retriever, "size", -1),
            self.top_k,
            len(hits),
            retrieval_s,
        )

        retrieved_memory: List[Dict[str, Any]] = []
        for h in hits:
            meta = h.metadata
            tids = meta.get("turn_ids") or []
            if tids:
                turn_id = f"{tids[0]}-{tids[-1]}" if len(tids) > 1 else str(tids[0])
            else:
                turn_id = ""
            principal_id = (meta.get("principal_ids") or [""])[0]
            role = (meta.get("roles") or [""])[0]
            refs = meta.get("record_refs", []) or []
            retrieved_memory.append(
                {
                    "record_id": refs[0] if refs else h.chunk_id,
                    "turn_id": turn_id,
                    "principal_id": principal_id,
                    "role": role,
                    "text": h.text,
                    "record_refs": refs,
                    "score": h.score,
                }
            )

        out = self._run_llm(checkpoint=checkpoint, retrieved_memory=retrieved_memory)
        out["retrieval_s"] = retrieval_s
        if q_usage:
            out["embedding_query_usage"] = q_usage
        if self.retrieval_backend == "embedding":
            out["embedding_doc_usage_accum"] = dict(self._embed_usage_accum)
        return out
