from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from ..models import EvalSingleResult, SuiteEvalResult, SuiteReport

if TYPE_CHECKING:
    from ..middleware.monitor.monitor import Monitor


class EvalSuite:
    def __init__(self, name: str = ""):
        self.name = name
        self._evals: List[dict] = []

    def add(self, name: str, eval_fn: Callable[[], EvalSingleResult]) -> EvalSuite:
        self._evals.append({"name": name, "fn": eval_fn})
        return self

    @classmethod
    def from_monitor(
        cls,
        monitor: "Monitor",
        embedder: Callable[[str], List[float]],
        corpus: Dict[str, str],
        top: Optional[int] = 10,
        use_problematic: bool = True,
        min_rank: float = 5.0,
        max_score: float = 0.7,
        k: int = 3,
        name: str = "from_monitor",
    ) -> "EvalSuite":
        """
        Build an EvalSuite from production monitoring data.

        corpus is a dict of {doc_id: text}. For each cluster, the expected doc is
        retrieved_doc_ids[0] from the most recent event that has retrieval data.
        Clusters with no retrieval data are skipped.
        """
        from ..fns.embedding import embedding_eval
        from ..metrics import Metrics

        suite = cls(name=name)

        clusters = (
            monitor.problematic_clusters(min_rank=min_rank, max_score=max_score, top=top)
            if use_problematic
            else monitor.clusters(top=top)
        )

        corpus_ids = list(corpus.keys())
        corpus_vecs = [embedder(corpus[doc_id]) for doc_id in corpus_ids]

        for cluster in clusters:
            events = monitor.events(cluster_id=cluster.cluster_id, has_retrieved_docs=True)
            if not events:
                continue
            top_doc_id = events[0].retrieved_doc_ids[0]
            if top_doc_id not in corpus:
                continue

            query_vec = embedder(cluster.representative_query)
            expected_vec = embedder(corpus[top_doc_id])

            suite.add(
                name=cluster.representative_query[:50],
                eval_fn=embedding_eval(
                    query_vec=query_vec,
                    expected_vec=expected_vec,
                    corpus_vecs=corpus_vecs,
                    k=k,
                ),
            )

        return suite

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
