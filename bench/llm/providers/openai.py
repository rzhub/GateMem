from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from ..router import LLMError, backoff_sleep, get_env_api_key
from ..types import LLMConfig, LLMCallResult

_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_TEXT_VERBOSITIES = {"low", "medium", "high"}


class OpenAIClient:
    """OpenAI Responses API client (REST)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_base = (config.api_base or "https://api.openai.com/v1").rstrip("/")
        env = config.api_key_env or "OPENAI_API_KEY"
        self.api_key = get_env_api_key(env)

    def complete_result(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        json_schema_name: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> LLMCallResult:
        url = f"{self.api_base}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        model = self.config.model
        effort = _normalize_openai_reasoning_effort(reasoning_effort or self.config.reasoning_effort)
        verbosity = _normalize_text_verbosity(text_verbosity or self.config.text_verbosity)
        payload: Dict[str, Any] = {
            "model": model,
            "input": user_prompt,
            "max_output_tokens": int(self.config.max_output_tokens),
            "text": {"format": {"type": "text"}},
        }
        if _should_send_openai_temperature(model=model, reasoning_effort=effort):
            payload["temperature"] = float(self.config.temperature)
        if json_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": (json_schema_name or "structured_output"),
                    "strict": True,
                    "schema": json_schema,
                }
            }
        if verbosity is not None:
            payload.setdefault("text", {}).update({"verbosity": verbosity})
        if effort is not None:
            payload["reasoning"] = {"effort": effort}
        if system_prompt.strip():
            payload["instructions"] = system_prompt
        if provider_options:
            payload.update(provider_options)

        last_err: Optional[str] = None
        for attempt in range(max(1, int(self.config.max_retries))):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.config.timeout_s,
                )
            except Exception as e:
                last_err = f"network_error: {e}"
                backoff_sleep(attempt)
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"http_{resp.status_code}: {resp.text[:500]}"
                backoff_sleep(attempt)
                continue

            if resp.status_code < 200 or resp.status_code >= 300:
                raise LLMError(f"OpenAI API error {resp.status_code}: {resp.text}")

            data = resp.json()
            text = _extract_openai_output_text(data)
            if text is None:
                status = data.get("status") if isinstance(data, dict) else None
                incomplete = data.get("incomplete_details") if isinstance(data, dict) else None
                err = data.get("error") if isinstance(data, dict) else None
                raise LLMError(
                    f"OpenAI API response missing text. status={status} incomplete_details={incomplete} error={err} keys={list(data.keys())}"
                )
            usage = _extract_openai_usage(data)
            return LLMCallResult(
                text=text,
                provider="openai",
                model=model,
                latency_s=0.0,
                usage=usage,
                raw=data if isinstance(data, dict) else None,
            )

        raise LLMError(f"OpenAI API failed after retries. Last error: {last_err}")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.complete_result(system_prompt=system_prompt, user_prompt=user_prompt).text


def _is_openai_gpt5_family(model: str) -> bool:
    m = (model or "").lower().strip()
    return m.startswith("gpt-5")


def _should_send_openai_temperature(*, model: str, reasoning_effort: Optional[str]) -> bool:
    if not _is_openai_gpt5_family(model):
        return True
    m = (model or "").lower().strip()
    if m.startswith(("gpt-5.2", "gpt-5.4")) and reasoning_effort == "none":
        return True
    return False


def _normalize_openai_reasoning_effort(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if s not in _REASONING_EFFORTS:
        raise LLMError(f"Unsupported OpenAI reasoning effort: {v}")
    return s


def _normalize_text_verbosity(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if s not in _TEXT_VERBOSITIES:
        raise LLMError(f"Unsupported text verbosity: {v}")
    return s


def _extract_openai_output_text(data: Dict[str, Any]) -> Optional[str]:
    out = data.get("output")
    if not isinstance(out, list):
        return None

    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") in ("output_text", "text"):
                t = c.get("text")
                if isinstance(t, str):
                    parts.append(t)
    if not parts:
        return None
    return "".join(parts).strip()


def _extract_openai_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        inp_i = int(usage.get("input_tokens") or 0)
        out_i = int(usage.get("output_tokens") or 0)
        tot_i = int(usage.get("total_tokens") or (inp_i + out_i))
        details = usage.get("output_tokens_details") or {}
        reasoning_i = int((details or {}).get("reasoning_tokens") or 0)
        return {
            "input_tokens": inp_i,
            "output_tokens": out_i,
            "total_tokens": tot_i,
            "reasoning_tokens": reasoning_i,
        }
    except Exception:
        return None
