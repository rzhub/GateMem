from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from .types import LLMConfig, LLMCallResult


class LLMError(RuntimeError):
    pass


class LLMRouter:
    """Thin wrapper that routes calls to the selected provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        provider = (config.provider or "stub").lower().strip()
        self.provider = provider

        if provider == "openai":
            from .providers.openai import OpenAIClient

            self._client = OpenAIClient(config)
        elif provider in ("anthropic", "claude"):
            from .providers.anthropic import AnthropicClient

            self._client = AnthropicClient(config)
        elif provider in ("gemini", "google"):
            from .providers.gemini import GeminiClient

            self._client = GeminiClient(config)
        elif provider == "deepseek":
            from .providers.deepseek import DeepSeekClient

            self._client = DeepSeekClient(config)
        elif provider in ("llama", "nvidia"):
            from .providers.llama import LlamaClient

            self._client = LlamaClient(config)
        elif provider == "stub":
            from .providers.stub import StubClient

            self._client = StubClient(config)
        else:
            raise ValueError(f"Unknown provider: {provider}")

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
        """Return normalized output + token usage + latency.

        The router forwards a small, provider-agnostic capability surface used by
        the benchmark: structured outputs, reasoning-effort hints, verbosity
        hints, and optional provider-specific knobs.
        """

        if self.config.merge_system_into_user:
            merged = (system_prompt.strip() + "\n\n" + user_prompt.strip()).strip()
            system_prompt = ""
            user_prompt = merged

        t0 = time.perf_counter()
        client_kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "json_schema": json_schema,
            "json_schema_name": json_schema_name,
            "reasoning_effort": reasoning_effort,
            "text_verbosity": text_verbosity,
            "provider_options": provider_options,
        }
        if hasattr(self._client, "complete_result"):
            res: LLMCallResult = self._client.complete_result(**client_kwargs)
        else:
            txt = self._client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            res = LLMCallResult(
                text=txt,
                provider=self.provider,
                model=self.config.model,
                latency_s=0.0,
                usage=None,
                raw=None,
            )

        res.latency_s = max(0.0, time.perf_counter() - t0)
        res.provider = res.provider or self.provider
        res.model = res.model or self.config.model
        return res

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        json_schema_name: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Backward compatible text-only completion."""
        return self.complete_result(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            json_schema_name=json_schema_name,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
            provider_options=provider_options,
        ).text

    def debug_config(self) -> Dict[str, Any]:
        return asdict(self.config)


def get_env_api_key(env_name: str) -> str:
    import os

    key = os.getenv(env_name)
    if not key:
        raise LLMError(
            f"Missing API key. Please set environment variable {env_name} before running."
        )
    return key


def backoff_sleep(attempt: int) -> None:
    sec = min(10.0, (2.0 ** attempt))
    time.sleep(sec)
