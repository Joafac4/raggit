from __future__ import annotations

import math
from datetime import datetime
from typing import List, Literal

from .embedder import Embedder
from .models import EmbeddingPair, EvalResult, EvalRun


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class RaggitEval:
    def __init__(self, pairs: List[EmbeddingPair]):
        self.pairs = pairs

    def compare(
        self,
        model_a: Embedder,
        model_b: Embedder,
        mode: Literal["auto", "human", "both"] = "auto",
    ) -> EvalRun:
        """Compare two embedding models on the provided pairs.

        mode="auto"  — compare using cosine similarity only (MVP)
        mode="human" — not yet implemented
        mode="both"  — not yet implemented
        """
        if mode != "auto":
            raise NotImplementedError(f"mode={mode!r} is not implemented in the MVP. Use mode='auto'.")

        results_a: List[EvalResult] = []
        results_b: List[EvalResult] = []

        for pair in self.pairs:
            ts = datetime.now()

            sim_a = _cosine_similarity(
                model_a.embed(pair.query),
                model_a.embed(pair.relevant_doc),
            )
            sim_b = _cosine_similarity(
                model_b.embed(pair.query),
                model_b.embed(pair.relevant_doc),
            )

            results_a.append(EvalResult(
                model_name=model_a.model_name,
                query=pair.query,
                relevant_doc=pair.relevant_doc,
                similarity_score=sim_a,
                timestamp=ts,
            ))
            results_b.append(EvalResult(
                model_name=model_b.model_name,
                query=pair.query,
                relevant_doc=pair.relevant_doc,
                similarity_score=sim_b,
                timestamp=ts,
            ))

        wins_a = sum(
            1 for ra, rb in zip(results_a, results_b)
            if ra.similarity_score > rb.similarity_score
        )
        wins_b = sum(
            1 for ra, rb in zip(results_a, results_b)
            if rb.similarity_score > ra.similarity_score
        )

        if wins_a > wins_b:
            winner = model_a.model_name
        elif wins_b > wins_a:
            winner = model_b.model_name
        else:
            winner = None

        from .report import Report  # local import to avoid circular deps at module level

        run = EvalRun(
            model_a=model_a.model_name,
            model_b=model_b.model_name,
            pairs=self.pairs,
            results_a=results_a,
            results_b=results_b,
            winner=winner,
            created_at=datetime.now(),
        )
        run.report = Report(run)
        return run
