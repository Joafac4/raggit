from __future__ import annotations

from typing import List, Optional

from ..base import BaseEval
from ..corpus import Corpus
from ...metrics import Metrics, _MetricFn
from ...models import EvalSingleResult

_VECTOR_MATCH_THRESHOLD = 0.999


class EmbeddingEval(BaseEval):
    def __init__(
        self,
        query: str,
        expected_doc: str,
        corpus: Corpus,
        k: int = 3,
        metric: _MetricFn = Metrics.cosine_similarity,
        name: str = "",
    ):
        self.query = query
        self.expected_doc = expected_doc
        self.corpus = corpus
        self.k = k
        self.metric = metric
        self.name = name or f"{corpus.embedder.model_name}: {query[:40]}"

    def run(self) -> EvalSingleResult:
        query_vec = self.corpus.embedder.embed(self.query)
        expected_vec = self.corpus.embedder.embed(self.expected_doc)

        scored = [
            (i, self.metric(query_vec, doc_vec))
            for i, doc_vec in enumerate(self.corpus.vecs)
        ]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        rank: Optional[int] = None
        for position, (idx, _) in enumerate(ranked, start=1):
            if Metrics.cosine_similarity(expected_vec, self.corpus.vecs[idx]) >= _VECTOR_MATCH_THRESHOLD:
                rank = position
                break

        return EvalSingleResult(
            hit=rank is not None and rank <= self.k,
            rank=rank,
            score=self.metric(query_vec, expected_vec),
            k=self.k,
            metric_name=self.metric.__name__,
        )
