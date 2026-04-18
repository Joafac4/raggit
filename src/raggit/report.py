from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from .models import EvalRun


def _utf8_console() -> Console:
    """Return a Console that always writes UTF-8, even on Windows cp1252 terminals."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass
    return Console(legacy_windows=False)


class Report:
    def __init__(self, run: EvalRun):
        self.run = run

    def show(self) -> None:
        run = self.run
        console = _utf8_console()

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", expand=False)
        table.add_column("Query", style="white", min_width=22)
        table.add_column(f"{run.model_a[:20]}", justify="center", min_width=13)
        table.add_column(f"{run.model_b[:20]}", justify="center", min_width=13)

        for ra, rb in zip(run.results_a, run.results_b):
            q = ra.query
            query_display = f'"{q[:20]}..."' if len(q) > 20 else f'"{q}"'

            def fmt(r) -> str:
                score = f"{r.metric_score:.2f}"
                rank  = f"#{r.rank}" if r.rank else "-"
                check = " ✓" if r.hit else ""
                style = "bold green" if r.hit else ""
                cell  = f"{score}  {rank}{check}"
                return f"[{style}]{cell}[/{style}]" if style else cell

            table.add_row(query_display, fmt(ra), fmt(rb))

        total = len(run.pairs)
        avg_a = sum(r.metric_score for r in run.results_a) / total if total else 0.0
        avg_b = sum(r.metric_score for r in run.results_b) / total if total else 0.0
        hits_a = sum(1 for r in run.results_a if r.hit)
        hits_b = sum(1 for r in run.results_b if r.hit)

        if run.winner == run.model_a:
            winner_str = f"[bold green]{run.model_a}[/bold green]"
        elif run.winner == run.model_b:
            winner_str = f"[bold green]{run.model_b}[/bold green]"
        else:
            winner_str = "[yellow]Tie[/yellow]"

        console.print()
        console.rule("[bold]Raggit Eval Report[/bold]")
        console.print(f"  Run ID : [dim]{run.run_id[:8]}[/dim]")
        console.print(f"  Date   : {run.created_at.strftime('%Y-%m-%d %H:%M')}")
        console.print(f"  Model A: [cyan]{run.model_a}[/cyan]")
        console.print(f"  Model B: [cyan]{run.model_b}[/cyan]")
        console.print()
        console.print(table)
        console.print()
        console.print(f"  Winner        : {winner_str}")
        console.print(f"  Avg score  A  : {avg_a:.3f}  |  hits: {hits_a}/{total}")
        console.print(f"  Avg score  B  : {avg_b:.3f}  |  hits: {hits_b}/{total}")
        console.rule()
