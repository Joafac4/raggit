from __future__ import annotations

from typing import Callable, List, Optional

from ..evaluation.corpus import Corpus
from ..metrics import Metrics, _MetricFn
from ..models import EvalSingleResult

_VECTOR_MATCH_THRESHOLD = 0.999


def embedding_eval(
    query: str,
    expected: str,
    corpus: Corpus,
    k: int = 3,
    metric: _MetricFn = Metrics.cosine_similarity,
) -> Callable[[], EvalSingleResult]:
    """
    Factory that returns an eval function for embedding retrieval.

    The corpus is shared — embeddings are pre-computed once in Corpus and
    reused across every eval that references the same Corpus instance.

    Usage:
        corpus = Corpus(docs=my_docs, embedder=embedder)

        suite.add("activate", embedding_eval("How to activate?", "To activate...", corpus))
        suite.add("reset",    embedding_eval("Reset password",   "Visit login...", corpus))
    """
    def _run() -> EvalSingleResult:
        query_vec    = corpus.embedder.embed(query)
        expected_vec = corpus.embedder.embed(expected)

        scored = [
            (i, metric(query_vec, doc_vec))
            for i, doc_vec in enumerate(corpus.vecs)
        ]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        rank: Optional[int] = None
        for position, (idx, _) in enumerate(ranked, start=1):
            if Metrics.cosine_similarity(expected_vec, corpus.vecs[idx]) >= _VECTOR_MATCH_THRESHOLD:
                rank = position
                break

        return EvalSingleResult(
            passed=rank is not None and rank <= k,
            score=metric(query_vec, expected_vec),
            rank=rank,
            metric_name=metric.__name__,
        )

    return _run
