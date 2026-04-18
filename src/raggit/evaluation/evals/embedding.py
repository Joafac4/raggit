from __future__ import annotations

from typing import List

from ..base import BaseEval
from ..engine import Evaluation, _MetricFn
from ...embedder import Embedder
from ...metrics import Metrics
from ...models import EvalSingleResult


class EmbeddingEval(BaseEval):
    def __init__(
        self,
        query: str,
        expected_doc: str,
        corpus: List[str],
        embedder: Embedder,
        k: int = 3,
        metric: _MetricFn = Metrics.cosine_similarity,
        name: str = "",
    ):
        self.query = query
        self.expected_doc = expected_doc
        self.corpus = corpus
        self.embedder = embedder
        self.k = k
        self.metric = metric
        self.name = name or f"{embedder.model_name}: {query[:40]}"

    def run(self) -> EvalSingleResult:
        corpus_vecs = [self.embedder.embed(doc) for doc in self.corpus]
        query_vec = self.embedder.embed(self.query)
        expected_vec = self.embedder.embed(self.expected_doc)

        return Evaluation(corpus_vecs=corpus_vecs).eval(
            query_vec=query_vec,
            expected_vec=expected_vec,
            k=self.k,
            metric=self.metric,
        )
