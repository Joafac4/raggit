from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import EvalSingleResult


class BaseEval(ABC):
    name: str  # human-readable name shown in the report

    @abstractmethod
    def run(self) -> EvalSingleResult:
        ...
