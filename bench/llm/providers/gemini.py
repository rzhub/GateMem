from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from ..router import LLMError, backoff_sleep, get_env_api_key
from ..types import LLMConfig, LLMCallResult

_GEMINI_THINKING_LEVELS = {"minimal", "low", "medium", "high"}


class GeminiClient:
    """Gemini generateContent API client (REST)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_base = (
            config.api_base or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        env = config.api_key_env or "GEMINI_API_KEY"
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
        del json_schema_name, text_verbosity  # Gemini JSON mode does not require a schema name.
        model = self.config.model
        if not model.startswith("models/"):
            model = f"models/{model}"

        url = f"{self.api_base}/{model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        generation_config: Dict[str, Any] = {
            "temperature": float(self.config.temperature),
            "maxOutputTokens": int(self.config.max_output_tokens),
        }
        thinking_cfg = _build_gemini_thinking_config(self.config.model, reasoning_effort or self.config.reasoning_effort)
        if thinking_cfg:
            generation_config["thinkingConfig"] = thinking_cfg
        if json_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseJsonSchema"] = json_schema

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
        if system_prompt.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }
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
                raise LLMError(f"Gemini API error {resp.status_code}: {resp.text}")

            data = resp.json()
            text = _extract_gemini_text(data)
            if text is None:
                raise LLMError(
                    f"Gemini response missing text. Keys: {list(data.keys())}"
                )
            usage = _extract_gemini_usage(data)
            return LLMCallResult(
                text=text,
                provider="gemini",
                model=self.config.model,
                latency_s=0.0,
                usage=usage,
                raw=data if isinstance(data, dict) else None,
            )

        raise LLMError(f"Gemini API failed after retries. Last error: {last_err}")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.complete_result(system_prompt=system_prompt, user_prompt=user_prompt).text


def _normalize_gemini_reasoning_effort(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    mapping = {
        "none": "minimal",
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
    }
    if s not in mapping:
        raise LLMError(f"Unsupported Gemini reasoning effort: {v}")
    return mapping[s]


def _gemini_thinking_budget(level: str) -> int:
    return {
        "minimal": 0,
        "low": 256,
        "medium": 1024,
        "high": 2048,
    }[level]


def _build_gemini_thinking_config(model: str, effort: Optional[str]) -> Optional[Dict[str, Any]]:
    level = _normalize_gemini_reasoning_effort(effort)
    if level is None:
        return None
    m = (model or "").lower().strip()
    if m.startswith("gemini-3") or "gemini-3" in m:
        return {"thinkingLevel": level}
    return {"thinkingBudget": _gemini_thinking_budget(level)}


def _extract_gemini_text(data: Dict[str, Any]) -> Optional[str]:
    cands = data.get("candidates")
    if not isinstance(cands, list) or not cands:
        return None
    c0 = cands[0]
    if not isinstance(c0, dict):
        return None
    content = c0.get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return None
    texts: list[str] = []
    for p in parts:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            texts.append(p["text"])
    if not texts:
        return None
    return "".join(texts).strip()


def _extract_gemini_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    meta = data.get("usageMetadata")
    if not isinstance(meta, dict):
        return None
    try:
        inp = int(meta.get("promptTokenCount") or 0)
        out = int(meta.get("candidatesTokenCount") or 0)
        tot = int(meta.get("totalTokenCount") or (inp + out))
        reasoning = int(meta.get("thoughtsTokenCount") or 0)
        return {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": tot,
            "reasoning_tokens": reasoning,
        }
    except Exception:
        return None
