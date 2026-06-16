from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from ..router import LLMError, backoff_sleep, get_env_api_key
from ..types import LLMConfig, LLMCallResult


class OpenAICompatibleChatClient:
    """Generic OpenAI-compatible /chat/completions client.

    This is used for providers whose APIs follow the OpenAI chat-completions
    shape rather than the OpenAI Responses API used by bench.llm.providers.openai.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        provider_name: str,
        default_api_base: str,
        default_api_key_env: str,
        default_model: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.config = config
        self.provider_name = provider_name
        self.default_model = default_model
        self.api_base = (config.api_base or default_api_base).rstrip("/")
        env = config.api_key_env or default_api_key_env
        self.api_key = get_env_api_key(env)
        self.extra_headers = dict(extra_headers or {})

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
        del json_schema_name, reasoning_effort, text_verbosity

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        model = self.config.model or self.default_model
        if not model:
            raise LLMError(f"Missing model name for provider {self.provider_name}")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_output_tokens),
        }

        # Most OpenAI-compatible chat APIs support JSON-object mode, while full
        # JSON-schema support is provider-specific. The benchmark judge already
        # retries with plain prompting if strict structured requests fail.
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}

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
                raise LLMError(
                    f"{self.provider_name} API error {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            text = _extract_chat_completion_text(data)
            if text is None:
                raise LLMError(
                    f"{self.provider_name} response missing text. Keys: {list(data.keys())}"
                )
            usage = _extract_chat_completion_usage(data)
            return LLMCallResult(
                text=text,
                provider=self.provider_name,
                model=model,
                latency_s=0.0,
                usage=usage,
                raw=data if isinstance(data, dict) else None,
            )

        raise LLMError(f"{self.provider_name} API failed after retries. Last error: {last_err}")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.complete_result(system_prompt=system_prompt, user_prompt=user_prompt).text


def _extract_chat_completion_text(data: Dict[str, Any]) -> Optional[str]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    msg = first.get("message") or {}
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("content"), str):
                parts.append(item["content"])
        if parts:
            return "".join(parts).strip()
    return None


def _extract_chat_completion_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        tot = int(usage.get("total_tokens") or (inp + out))
        details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        reasoning = int((details or {}).get("reasoning_tokens") or 0)
        return {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": tot,
            "reasoning_tokens": reasoning,
        }
    except Exception:
        return None
