from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from ..router import LLMError, backoff_sleep, get_env_api_key
from ..types import LLMConfig, LLMCallResult

_ANTHROPIC_EFFORTS = {"low", "medium", "high", "max"}


class AnthropicClient:
    """Anthropic Messages API client (REST)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_base = (config.api_base or "https://api.anthropic.com/v1").rstrip("/")
        env = config.api_key_env or "ANTHROPIC_API_KEY"
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
        del json_schema_name, text_verbosity
        url = f"{self.api_base}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.config.anthropic_version,
            "content-type": "application/json",
        }

        model = self.config.model
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": int(self.config.max_output_tokens),
            "temperature": float(self.config.temperature),
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt.strip():
            payload["system"] = system_prompt

        effort = _normalize_anthropic_effort(reasoning_effort or self.config.reasoning_effort, model=model)
        output_config: Dict[str, Any] = {}
        if effort is not None:
            output_config["effort"] = effort
            payload["thinking"] = _build_anthropic_thinking(model=model, effort=effort)
        if json_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": json_schema,
            }
        if output_config:
            payload["output_config"] = output_config
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

            if resp.status_code in (429, 500, 502, 503, 504, 529):
                last_err = f"http_{resp.status_code}: {resp.text[:500]}"
                backoff_sleep(attempt)
                continue

            if resp.status_code < 200 or resp.status_code >= 300:
                raise LLMError(f"Anthropic API error {resp.status_code}: {resp.text}")

            data = resp.json()
            text = _extract_anthropic_text(data)
            if text is None:
                stop_reason = data.get("stop_reason") if isinstance(data, dict) else None
                raise LLMError(
                    f"Anthropic API response missing text. stop_reason={stop_reason} keys={list(data.keys()) if isinstance(data, dict) else []}"
                )
            usage = _extract_anthropic_usage(data)
            return LLMCallResult(
                text=text,
                provider="anthropic",
                model=self.config.model,
                latency_s=0.0,
                usage=usage,
                raw=data if isinstance(data, dict) else None,
            )

        raise LLMError(f"Anthropic API failed after retries. Last error: {last_err}")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.complete_result(system_prompt=system_prompt, user_prompt=user_prompt).text


def _is_anthropic_46_model(model: str) -> bool:
    m = (model or "").lower().replace('.', '-').strip()
    return "4-6" in m


def _normalize_anthropic_effort(v: Optional[str], *, model: str) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    mapping = {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max" if _is_anthropic_46_model(model) else "high",
    }
    if s not in mapping:
        raise LLMError(f"Unsupported Anthropic reasoning effort: {v}")
    return mapping[s]


def _build_anthropic_thinking(*, model: str, effort: str) -> Dict[str, Any]:
    if _is_anthropic_46_model(model):
        return {"type": "adaptive"}
    budget = {
        "low": 1024,
        "medium": 4096,
        "high": 8192,
        "max": 8192,
    }[effort]
    return {"type": "enabled", "budget_tokens": budget}


def _extract_anthropic_text(data: Dict[str, Any]) -> Optional[str]:
    content = data.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for blk in content:
        if not isinstance(blk, dict):
            continue
        if blk.get("type") == "text" and isinstance(blk.get("text"), str):
            parts.append(blk["text"])
            continue
        if blk.get("type") == "output_json":
            value = blk.get("json")
            try:
                parts.append(json.dumps(value, ensure_ascii=False))
            except Exception:
                pass
    if not parts:
        return None
    return "".join(parts).strip()


def _extract_anthropic_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        reasoning = int(usage.get("thinking_tokens") or 0)
        return {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "reasoning_tokens": reasoning,
        }
    except Exception:
        return None
