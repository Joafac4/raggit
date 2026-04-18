from __future__ import annotations

import math
from typing import Callable, Dict, List


class Metrics:
    def __init__(self) -> None:
        self._registry: Dict[str, Callable[[List[float], List[float]], float]] = {
            "cosine_similarity": self.cosine_similarity,
            "euclidean_similarity": self.euclidean_similarity,
            "dot_product": self.dot_product,
        }

    # ── Built-in metrics ─────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def euclidean_similarity(a: List[float], b: List[float]) -> float:
        return 1 / (1 + math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))

    @staticmethod
    def dot_product(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def compare(self, vec_a: List[float], vec_b: List[float], metric_name: str) -> float:
        if metric_name not in self._registry:
            raise ValueError(
                f"Unknown metric {metric_name!r}. "
                f"Available: {sorted(self._registry)}"
            )
        return self._registry[metric_name](vec_a, vec_b)

    # ── Extensibility ────────────────────────────────────────────────────────

    def register_metric(self, name: str, fn: Callable[[List[float], List[float]], float]) -> None:
        self._registry[name] = fn
