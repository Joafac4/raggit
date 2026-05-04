from __future__ import annotations

from typing import Callable, List

from ..models import EvalSingleResult, SuiteEvalResult, SuiteReport


class EvalSuite:
    def __init__(self, name: str = ""):
        self.name = name
        self._evals: List[dict] = []

    def add(self, name: str, eval_fn: Callable[[], EvalSingleResult]) -> EvalSuite:
        self._evals.append({"name": name, "fn": eval_fn})
        return self

    def run(self) -> SuiteReport:
        results = [
            SuiteEvalResult(name=e["name"], result=e["fn"]())
            for e in self._evals
        ]
        passed = sum(1 for r in results if r.result.passed)
        total = len(results)
        return SuiteReport(
            suite_name=self.name,
            results=results,
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total if total else 0.0,
        )
