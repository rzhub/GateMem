from __future__ import annotations

import json
from typing import Any, Dict

from ..types import LLMConfig, LLMCallResult


class StubClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return "{}"

    @staticmethod
    def _schema_default_value(spec: Dict[str, Any]):
        types = spec.get("type")
        if isinstance(types, list):
            if "null" in types:
                return None
            types = types[0] if types else None
        if types == "boolean":
            return False
        if types == "string":
            return ""
        if types == "array":
            return []
        if types == "object":
            props = spec.get("properties") or {}
            return {k: StubClient._schema_default_value(v or {}) for k, v in props.items()}
        if types in {"integer", "number"}:
            return 0
        return None

    def complete_result(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema=None,
        json_schema_name=None,
        reasoning_effort=None,
        text_verbosity=None,
        provider_options=None,
    ) -> LLMCallResult:
        del json_schema_name, reasoning_effort, text_verbosity, provider_options
        if isinstance(json_schema, dict):
            text = json.dumps(self._schema_default_value(json_schema), ensure_ascii=False)
        else:
            text = self.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        return LLMCallResult(
            text=text,
            provider="stub",
            model=self.config.model,
            latency_s=0.0,
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            raw=None,
        )
