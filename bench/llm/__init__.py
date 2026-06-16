"""LLM provider abstraction used by agents.

The benchmark can run in an offline deterministic mode ("stub") or call real
hosted model APIs (OpenAI / Anthropic / Gemini / DeepSeek / Llama).
"""

from .types import LLMConfig
from .router import LLMRouter, LLMError

__all__ = ["LLMConfig", "LLMRouter", "LLMError"]
