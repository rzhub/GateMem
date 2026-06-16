from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .base import BaseMemoryAgent, Turn, Checkpoint
from bench.retrieval import Chunker, TfidfRetriever, EmbeddingRetriever, LangChainFaissRetriever


class NaiveRAGAgent(BaseMemoryAgent):
    """Baseline: retrieval-augmented generation over chunks, no policy filtering."""

    def __init__(
        self,
        *,
        top_k: int = 5,
        llm_mode: str = "leaky",
        llm_router: Any = None,
        query_prompt_path: str | None = None,
        retrieval_backend: str = "tfidf",  # tfidf|embedding
        chunk_mode: str = "turn",  # turn|window|chars
        chunk_window_turns: int = 5,
        chunk_max_chars: int = 4000,
        embed_router: Optional[Any] = None,
        use_faiss: bool = False,
        embedding_impl: str = "native",  # native|langchain
        embed_provider: str = "openai",  # for langchain impl
        embed_model: str = "text-embedding-3-small",  # for langchain impl
        embed_api_base: Optional[str] = None,
        embed_api_key_env: Optional[str] = None,
        embed_device: str = "cpu",
        embed_batch_size: int = 16,
    ) -> None:
        super().__init__(
            top_k=top_k,
            llm_mode=llm_mode,
            llm_router=llm_router,
            query_prompt_path=query_prompt_path,
        )
        self.retrieval_backend = retrieval_backend
        self.embedding_impl = embedding_impl
        self._embed_router = embed_router
        self._embed_provider = embed_provider
        self._embed_model = embed_model
        self._embed_api_base = embed_api_base
        self._embed_api_key_env = embed_api_key_env
        self._embed_device = embed_device
        self._embed_batch_size = embed_batch_size
        self._use_faiss = use_faiss
        self.chunker = Chunker(mode=chunk_mode, window_turns=chunk_window_turns, max_chars=chunk_max_chars)

        if retrieval_backend == "embedding":
            if embedding_impl == "langchain":
                self.retriever = LangChainFaissRetriever(
                    provider=embed_provider,
                    model=embed_model,
                    logger=self.logger,
                    api_base=embed_api_base,
                    api_key_env=embed_api_key_env,
                    device=embed_device,
                    batch_size=embed_batch_size,
                )
            else:
                if embed_router is None:
                    raise ValueError("retrieval_backend=embedding with embedding_impl=native requires embed_router")
                self.retriever = EmbeddingRetriever(
                    embed_router=embed_router,
                    use_faiss=use_faiss,
                    show_progress=False,
                    logger=self.logger,
                )
        else:
            self.retriever = TfidfRetriever(logger=self.logger)

        self._embed_usage_accum: Dict[str, float] = {"input_tokens": 0, "total_tokens": 0}

    def reset(self, episode: Dict[str, Any]) -> None:
        super().reset(episode)
        self.chunker = Chunker(
            mode=self.chunker.mode,
            window_turns=self.chunker.window_turns,
            max_chars=self.chunker.max_chars,
        )
        if self.retrieval_backend == "embedding":
            # Recreate to clear state
            if self.embedding_impl == "langchain":
                self.retriever = LangChainFaissRetriever(
                    provider=self._embed_provider,
                    model=self._embed_model,
                    logger=self.logger,
                    api_base=self._embed_api_base,
                    api_key_env=self._embed_api_key_env,
                    device=self._embed_device,
                    batch_size=self._embed_batch_size,
                )
            else:
                self.retriever = EmbeddingRetriever(
                    embed_router=self._embed_router,
                    use_faiss=self._use_faiss,
                    logger=self.logger,
                )
        else:
            self.retriever = TfidfRetriever(logger=self.logger)
        self._embed_usage_accum = {"input_tokens": 0, "total_tokens": 0}

    def ingest(self, turn: Turn) -> None:
        tdict = {
            "turn_id": turn.turn_id,
            "speaker": {"principal_id": turn.speaker_principal_id, "role": turn.speaker_role},
            "text": turn.text,
            "record_refs": self._merge_record_refs(turn.record_refs, turn.text),
        }
        chunks = self.chunker.add_turn(tdict)
        for ch in chunks:
            meta = {
                "turn_ids": ch.turn_ids,
                "principal_ids": ch.principal_ids,
                "roles": ch.roles,
                "record_refs": self._merge_record_refs(ch.record_refs, ch.text),
            }
            if isinstance(self.retriever, EmbeddingRetriever):
                usage = self.retriever.add(ch.chunk_id, ch.text, meta)
                # accumulate embedding token usage if present
                it = usage.get("input_tokens")
                tt = usage.get("total_tokens")
                if isinstance(it, (int, float)):
                    self._embed_usage_accum["input_tokens"] += float(it)
                if isinstance(tt, (int, float)):
                    self._embed_usage_accum["total_tokens"] += float(tt)
            else:
                self.retriever.add(ch.chunk_id, ch.text, meta)  # type: ignore

    def query(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        # ensure pending chunk is included
        for ch in self.chunker.finalize():
            meta = {
                "turn_ids": ch.turn_ids,
                "principal_ids": ch.principal_ids,
                "roles": ch.roles,
                "record_refs": self._merge_record_refs(ch.record_refs, ch.text),
            }
            if isinstance(self.retriever, EmbeddingRetriever):
                self.retriever.add(ch.chunk_id, ch.text, meta)
            else:
                self.retriever.add(ch.chunk_id, ch.text, meta)  # type: ignore

        if isinstance(self.retriever, EmbeddingRetriever):
            hits, retrieval_s, q_usage = self.retriever.search(checkpoint.query_text, top_k=self.top_k)
        else:
            hits, retrieval_s = self.retriever.search(checkpoint.query_text, top_k=self.top_k)  # type: ignore
            q_usage = {}

        self.logger.info(
            "Retrieval: backend=%s docs=%d top_k=%d hits=%d (t=%0.3fs)",
            self.retrieval_backend,
            getattr(self.retriever, "size", -1),
            self.top_k,
            len(hits),
            retrieval_s,
        )

        retrieved_memory: List[Dict[str, Any]] = []
        for h in hits:
            meta = h.metadata
            tids = meta.get("turn_ids") or []
            if tids:
                turn_id = f"{tids[0]}-{tids[-1]}" if len(tids) > 1 else str(tids[0])
            else:
                turn_id = ""
            principal_id = (meta.get("principal_ids") or [""])[0]
            role = (meta.get("roles") or [""])[0]
            retrieved_memory.append(
                {
                    "record_id": h.chunk_id,
                    "turn_id": turn_id,
                    "principal_id": principal_id,
                    "role": role,
                    "text": h.text,
                    "record_refs": meta.get("record_refs", []),
                    "score": h.score,
                }
            )

        out = self._run_llm(checkpoint=checkpoint, retrieved_memory=retrieved_memory)
        out["retrieval_s"] = retrieval_s
        if q_usage:
            out["embedding_query_usage"] = q_usage
        if self.retrieval_backend == "embedding":
            out["embedding_doc_usage_accum"] = dict(self._embed_usage_accum)
        return out
