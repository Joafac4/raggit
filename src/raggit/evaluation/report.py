from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from ..models import SuiteReport


def _utf8_console() -> Console:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass
    return Console(legacy_windows=False)


def show(report: SuiteReport) -> None:
    console = _utf8_console()

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan", expand=False)
    table.add_column("Eval",  style="white", min_width=28)
    table.add_column("Passed", justify="center", min_width=7)
    table.add_column("Rank",   justify="center", min_width=6)
    table.add_column("Score",  justify="center", min_width=6)

    for ser in report.results:
        r = ser.result
        passed_cell = "[bold green]✓[/bold green]" if r.passed else "[red]✗[/red]"
        rank_cell   = str(r.rank) if r.rank is not None else "-"
        score_cell  = f"{r.score:.2f}" if r.score is not None else "-"
        table.add_row(ser.name[:40], passed_cell, rank_cell, score_cell)

    console.print()
    console.rule("[bold]Raggit Eval Suite[/bold]")
    console.print(f"  Suite : {report.suite_name}")
    console.print(f"  Date  : {report.created_at.strftime('%Y-%m-%d %H:%M')}")
    console.print()
    console.print(table)
    console.print(
        f"  Total: {report.total}  |  "
        f"Passed: [green]{report.passed}[/green]  |  "
        f"Failed: [red]{report.failed}[/red]  |  "
        f"Pass rate: {report.pass_rate * 100:.1f}%"
    )
    if report.aggregations:
        console.print()
        console.print("  Aggregations:")
        for agg in report.aggregations:
            console.print(f"    {agg.name:<16}: {agg.value:.3f}")
    console.rule()
