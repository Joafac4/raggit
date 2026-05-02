from __future__ import annotations

from typing import Any, Callable, List

from ..evaluation.measure import evaluate
from ..metrics import Metrics
from ..models import EvalSingleResult


def chunk_eval(
    document: Any,
    expected_vec: List[float],
    chunk_fn: Callable[[Any, float], List[Any]],
    embed_fn: Callable[[Any], List[float]],
    overlap: float = 0.0,
    threshold: float = 0.9,
) -> Callable[[], EvalSingleResult]:
    """
    Factory that returns an eval function for any chunking strategy.

    chunk_fn receives (document, overlap) and returns a list of chunks of any type.
    embed_fn converts each chunk to a vector for similarity comparison.
    Works with any modality: text, audio, video, image, etc.

    score = best chunk similarity to expected — how well the chunker preserved the
    expected content somewhere in the document. Useful for comparing chunking strategies.

    Usage:
        suite.add("no-overlap",  chunk_eval(doc, expected_vec, my_chunker, embed_fn, overlap=0.0))
        suite.add("25%-overlap", chunk_eval(doc, expected_vec, my_chunker, embed_fn, overlap=0.25))
    """
    def _run() -> EvalSingleResult:
        chunks = chunk_fn(document, overlap)
        chunk_vecs = [embed_fn(c) for c in chunks]
        ranked = sorted(
            chunk_vecs,
            key=lambda v: Metrics.cosine_similarity(v, expected_vec),
            reverse=True,
        )
        return evaluate(
            ranked,
            expected_vec,
            k=None,                       # any matching chunk counts as passing
            threshold=threshold,
            metric_name="chunk_coverage",
        )

    return _run
