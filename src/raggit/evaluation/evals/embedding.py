from __future__ import annotations

from ..base import BaseEval
from ..corpus import Corpus
from ..engine import Evaluation, _MetricFn
from ...metrics import Metrics
from ...models import EvalSingleResult


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

        return Evaluation(corpus_vecs=self.corpus.vecs).eval(
            query_vec=query_vec,
            expected_vec=expected_vec,
            k=self.k,
            metric=self.metric,
        )
