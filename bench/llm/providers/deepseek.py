from __future__ import annotations

from ..types import LLMConfig
from .openai_compatible import OpenAICompatibleChatClient


class DeepSeekClient(OpenAICompatibleChatClient):
    """DeepSeek OpenAI-compatible chat-completions client."""

    def __init__(self, config: LLMConfig):
        super().__init__(
            config,
            provider_name="deepseek",
            default_api_base="https://api.deepseek.com",
            default_api_key_env="DEEPSEEK_API_KEY",
            default_model="deepseek-v4-pro",
        )
