from __future__ import annotations

"""Optional LangChain+FAISS retriever and embedding helpers.

This module is only used when optional LangChain dependencies are installed.
It also exposes ``make_langchain_embedder`` so non-RAG agents can consume the
same LangChain embedding implementations through a router-like adapter.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class Retrieved:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


def make_langchain_embedder(
    *,
    provider: str,
    model: str,
    api_base: Optional[str] = None,
    api_key_env: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 16,
    normalize: bool = True,
):
    prov = (provider or "").lower()
    if prov == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore
        except Exception as e:
            raise RuntimeError("Install langchain-openai to use OpenAIEmbeddings") from e
        kwargs: Dict[str, Any] = {"model": model, "chunk_size": max(1, int(batch_size))}
        if api_base:
            kwargs["base_url"] = api_base
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key in env var {api_key_env}. Set it to use LangChain OpenAI embeddings."
                )
            kwargs["api_key"] = api_key
        return OpenAIEmbeddings(**kwargs)

    if prov == "hf":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        except Exception as e:
            raise RuntimeError("Install langchain-huggingface to use HuggingFaceEmbeddings") from e
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": bool(normalize),
                "batch_size": max(1, int(batch_size)),
            },
        )

    raise ValueError(f"Unknown embedding provider for langchain impl: {provider}")


class LangChainFaissRetriever:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        top_k: int = 5,
        logger: Any = None,
        normalize_L2: bool = True,
        api_base: Optional[str] = None,
        api_key_env: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.top_k = top_k
        self.logger = logger
        self.normalize_L2 = normalize_L2
        self.api_base = api_base
        self.api_key_env = api_key_env
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        try:
            from langchain_core.documents import Document  # type: ignore
            from langchain_community.vectorstores import FAISS  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "LangChain FAISS retriever requires `langchain-core` and `langchain-community`."
            ) from e

        self.Document = Document
        self.FAISS = FAISS

        self._embedder = make_langchain_embedder(
            provider=self.provider,
            model=self.model,
            api_base=self.api_base,
            api_key_env=self.api_key_env,
            device=self.device,
            batch_size=self.batch_size,
            normalize=self.normalize,
        )
        self._store = None
        self._docs: List[Any] = []

    def add(self, chunk_id: str, text: str, metadata: Dict[str, Any]) -> None:
        doc = self.Document(page_content=text, metadata={**metadata, "chunk_id": chunk_id})
        self._docs.append(doc)
        if self._store is None:
            self._store = self.FAISS.from_documents([doc], self._embedder, normalize_L2=self.normalize_L2)
        else:
            self._store.add_documents([doc])

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        allow_chunk_ids: Optional[Set[str]] = None,
    ):
        import time

        t0 = time.perf_counter()
        if self._store is None:
            return [], time.perf_counter() - t0, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        k = int(top_k or self.top_k)
        rows = self._store.similarity_search_with_score(query, k=max(k * 5, k))

        out: List[Retrieved] = []
        for doc, score in rows:
            cid = str(doc.metadata.get("chunk_id", ""))
            if allow_chunk_ids is not None and cid not in allow_chunk_ids:
                continue
            out.append(
                Retrieved(
                    chunk_id=cid,
                    score=float(-score),
                    text=doc.page_content,
                    metadata={k: v for k, v in dict(doc.metadata).items() if k != "chunk_id"},
                )
            )
            if len(out) >= k:
                break

        usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        return out, time.perf_counter() - t0, usage

    @property
    def size(self) -> int:
        return len(self._docs)
