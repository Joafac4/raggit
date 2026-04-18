from __future__ import annotations

from typing import List

from .base import BaseEval
from ..models import SuiteEvalResult, SuiteReport


class EvalSuite:
    def __init__(self, name: str = ""):
        self.name = name
        self.evals: List[BaseEval] = []

    def add(self, eval: BaseEval) -> EvalSuite:
        self.evals.append(eval)
        return self

    def run(self) -> SuiteReport:
        results = [
            SuiteEvalResult(eval_name=e.name, result=e.run())
            for e in self.evals
        ]

        passed = sum(1 for r in results if r.result.hit)
        total  = len(results)

        return SuiteReport(
            suite_name=self.name,
            results=results,
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total if total else 0.0,
        )
