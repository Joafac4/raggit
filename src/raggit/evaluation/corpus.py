from __future__ import annotations

from typing import Any, List

from ..embedder import Embedder


class Corpus:
    def __init__(self, docs: List[Any], embedder: Embedder):
        self.docs = docs
        self.embedder = embedder
        self.vecs: List[List[float]] = [embedder.embed(doc) for doc in docs]
