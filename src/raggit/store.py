from __future__ import annotations

from pathlib import Path
from typing import List

from .models import SuiteReport


class RaggitStore:
    def __init__(self, path: str = ".raggit"):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

    def save(self, report: SuiteReport) -> None:
        file_path = self.path / f"{report.created_at.strftime('%Y%m%dT%H%M%S')}_{report.suite_name or 'suite'}.json"
        file_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    def load(self, filename: str) -> SuiteReport:
        file_path = self.path / filename
        return SuiteReport.model_validate_json(file_path.read_text(encoding="utf-8"))

    def list_reports(self) -> List[SuiteReport]:
        files = sorted(self.path.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return [SuiteReport.model_validate_json(f.read_text(encoding="utf-8")) for f in files]
