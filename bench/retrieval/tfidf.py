from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Retrieved:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class TfidfRetriever:
    """TF-IDF cosine retriever over chunks.

    This is a deterministic, offline retriever used for quick iteration.
    """

    def __init__(self, *, logger: Any = None) -> None:
        self._chunks: List[Tuple[str, str, Dict[str, Any]]] = []  # (id, text, meta)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._mat = None
        self._dirty = True
        self.logger = logger

    def add(self, chunk_id: str, text: str, metadata: Dict[str, Any]) -> None:
        self._chunks.append((chunk_id, text, metadata))
        self._dirty = True

    def _build(self) -> None:
        texts = [t for _, t, _ in self._chunks]
        if not texts:
            self._mat = None
            self._dirty = False
            return
        if self.logger:
            self.logger.info("Building TF-IDF index: docs=%d", len(texts))
        self._mat = self._vectorizer.fit_transform(texts)
        self._dirty = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        allow_chunk_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[Retrieved], float]:
        t0 = time.perf_counter()
        if self._dirty:
            self._build()
        if self._mat is None or not self._chunks:
            return [], time.perf_counter() - t0

        q = self._vectorizer.transform([query])
        # cosine similarity since tfidf vectors are L2-normalized
        sims = (self._mat @ q.T).toarray().reshape(-1)

        idxs = np.argsort(-sims)
        out: List[Retrieved] = []
        for i in idxs:
            if sims[i] <= 0:
                break
            cid, text, meta = self._chunks[i]
            if allow_chunk_ids is not None and cid not in allow_chunk_ids:
                continue
            out.append(Retrieved(chunk_id=cid, score=float(sims[i]), text=text, metadata=meta))
            if len(out) >= top_k:
                break

        return out, time.perf_counter() - t0

    @property
    def size(self) -> int:
        return len(self._chunks)
