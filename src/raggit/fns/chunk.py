from __future__ import annotations

from typing import Any, Callable, List, Optional

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

    Usage:
        suite.add("no-overlap",  chunk_eval(doc, expected_vec, my_chunker, embed_fn, overlap=0.0))
        suite.add("25%-overlap", chunk_eval(doc, expected_vec, my_chunker, embed_fn, overlap=0.25))
        suite.add("50%-overlap", chunk_eval(doc, expected_vec, my_chunker, embed_fn, overlap=0.5))
    """
    def _run() -> EvalSingleResult:
        chunks = chunk_fn(document, overlap)

        rank: Optional[int] = None
        best_score = 0.0
        for position, chunk in enumerate(chunks, start=1):
            sim = Metrics.cosine_similarity(embed_fn(chunk), expected_vec)
            if sim > best_score:
                best_score = sim
            if sim >= threshold and rank is None:
                rank = position

        return EvalSingleResult(
            passed=rank is not None,
            score=best_score,
            rank=rank,
            metric_name="chunk_coverage",
        )

    return _run
