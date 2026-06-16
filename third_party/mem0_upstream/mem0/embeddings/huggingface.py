import logging
import threading
from typing import Dict, Literal, Optional, Tuple

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class HuggingFaceEmbedding(EmbeddingBase):
    """HuggingFace embedder with process-wide model cache.

    Upstream Mem0 creates one embedder per ``Memory`` instance. In this benchmark we may
    construct multiple episode workers in parallel, and each worker can initialize the same
    local ``SentenceTransformer`` model concurrently. With some transformer / torch version
    combinations this concurrent initialization can fail with a ``meta tensor`` error when the
    model is moved onto the target device.

    To keep episode-level parallelism while avoiding concurrent local model construction, we
    cache local ``SentenceTransformer`` instances process-wide and guard first-time model
    initialization with a lock. Remote TEI / OpenAI-compatible paths are unchanged.
    """

    _MODEL_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], SentenceTransformer] = {}
    _CACHE_LOCK = threading.Lock()

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        if config.huggingface_base_url:
            self.client = OpenAI(base_url=config.huggingface_base_url)
            self.config.model = self.config.model or "tei"
        else:
            self.config.model = self.config.model or "multi-qa-MiniLM-L6-cos-v1"
            model_kwargs = dict(self.config.model_kwargs or {})
            self.model = self._get_or_create_local_model(self.config.model, model_kwargs)
            self.config.embedding_dims = self.config.embedding_dims or self.model.get_sentence_embedding_dimension()

    @classmethod
    def _cache_key(cls, model_name: str, model_kwargs: Dict[str, object]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        # ``SentenceTransformer`` kwargs are tiny in our usage (mostly ``device``), so a stable
        # stringified tuple key is enough here and keeps the cache implementation simple.
        norm_items = tuple(sorted((str(k), str(v)) for k, v in (model_kwargs or {}).items()))
        return (str(model_name), norm_items)

    @classmethod
    def _get_or_create_local_model(cls, model_name: str, model_kwargs: Dict[str, object]) -> SentenceTransformer:
        key = cls._cache_key(model_name, model_kwargs)
        cached = cls._MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        with cls._CACHE_LOCK:
            cached = cls._MODEL_CACHE.get(key)
            if cached is not None:
                return cached
            model = SentenceTransformer(model_name, **model_kwargs)
            cls._MODEL_CACHE[key] = model
            return model

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Hugging Face.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        if self.config.huggingface_base_url:
            return self.client.embeddings.create(
                input=text, model=self.config.model, **self.config.model_kwargs
            ).data[0].embedding
        else:
            return self.model.encode(text, convert_to_numpy=True).tolist()
