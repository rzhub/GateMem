from __future__ import annotations

from ..types import LLMConfig
from .openai_compatible import OpenAICompatibleChatClient


class LlamaClient(OpenAICompatibleChatClient):
    """Hosted Llama OpenAI-compatible chat-completions client.

    The default endpoint uses NVIDIA NIM for Llama 4 Maverick. Override
    --api_base and --api_key_env to use another OpenAI-compatible Llama host.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(
            config,
            provider_name="llama",
            default_api_base="https://integrate.api.nvidia.com/v1",
            default_api_key_env="NVIDIA_API_KEY",
            default_model="meta/llama-4-maverick-17b-128e-instruct",
        )
