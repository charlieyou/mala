"""Logs subcommand for mala CLI: search and inspect run logs."""

from __future__ import annotations

from typing import Annotated

import typer

logs_app = typer.Typer(name="logs", help="Search and inspect mala run logs")


@logs_app.command()
def list(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
    all_runs: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Show all runs (not just recent)",
        ),
    ] = False,
) -> None:
    """List recent mala runs."""
    raise NotImplementedError("Not implemented yet")


@logs_app.command()
def sessions(
    issue: Annotated[
        str | None,
        typer.Option(
            "--issue",
            help="Filter by issue ID",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
    all_sessions: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Show all sessions (not just recent)",
        ),
    ] = False,
) -> None:
    """List Claude sessions from mala runs."""
    raise NotImplementedError("Not implemented yet")


@logs_app.command()
def show(
    run_id: Annotated[
        str,
        typer.Argument(
            help="Run ID to show details for",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
) -> None:
    """Show details for a specific run."""
    raise NotImplementedError("Not implemented yet")
