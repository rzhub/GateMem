from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    """Configuration for an LLM backend.

    provider:
        - "stub"       : deterministic fake model (no network)
        - "openai"     : OpenAI Responses API
        - "anthropic"  : Claude Messages API
        - "gemini"     : Google Gemini generateContent API
        - "deepseek"   : DeepSeek OpenAI-compatible Chat Completions API
        - "llama"      : Hosted Llama OpenAI-compatible Chat Completions API

    api_key_env:
        Environment variable name to read the API key from.
        If omitted, a provider-specific default is used.
    """

    provider: str = "stub"
    model: str = "gpt-4o-mini"

    temperature: float = 0.2
    max_output_tokens: int = 256

    # Cross-provider reasoning/output controls. Providers may map these
    # differently depending on model support.
    reasoning_effort: Optional[str] = None
    text_verbosity: Optional[str] = None

    timeout_s: float = 60.0
    max_retries: int = 3

    # Optional overrides
    api_base: Optional[str] = None

    # Provider-specific knobs
    api_key_env: Optional[str] = None
    anthropic_version: str = "2023-06-01"

    # If True, we will ignore provider-specific system fields and just merge
    # system_prompt + user_prompt into a single user message.
    merge_system_into_user: bool = False


@dataclass
class LLMCallResult:
    """Normalized result returned by all providers."""

    text: str
    provider: str
    model: str
    latency_s: float
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None
