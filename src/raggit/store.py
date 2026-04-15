from __future__ import annotations

from pathlib import Path
from typing import List

from .models import EvalRun


class RaggitStore:
    def __init__(self, path: str = ".raggit"):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

    def save_run(self, run: EvalRun) -> None:
        file_path = self.path / f"{run.run_id}.json"
        file_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    def load_run(self, run_id: str) -> EvalRun:
        file_path = self.path / f"{run_id}.json"
        return EvalRun.model_validate_json(file_path.read_text(encoding="utf-8"))

    def list_runs(self) -> List[EvalRun]:
        files = sorted(self.path.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return [EvalRun.model_validate_json(f.read_text(encoding="utf-8")) for f in files]
