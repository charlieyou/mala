"""Tests for the canonical validation wrapper helpers."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from src.domain.validation.spec import CommandKind, ValidationCommand
from src.domain.validation_wrapper import (
    build_canonical_wrapper,
    escape_for_bash_lc,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "command",
    [
        "uvx ruff check .",
        'bash -c "echo it\'s fine"',
        "FOO=bar cmd arg1 arg2",
        "a | b",
        "a; b",
        r"echo \backslash",
        'echo "double quotes"',
        "''",
        "''nested''",
    ],
)
def test_escape_for_bash_lc_round_trips_through_single_quotes(command: str) -> None:
    """Wrapping the escape output in single quotes round-trips through bash parsing."""
    escaped = escape_for_bash_lc(command)
    parsed = shlex.split(f"'{escaped}'", posix=True)
    assert parsed == [command]


def test_escape_for_bash_lc_preserves_command_with_no_single_quotes() -> None:
    assert escape_for_bash_lc("uvx ruff check .") == "uvx ruff check ."


def test_escape_for_bash_lc_replaces_single_quotes_with_shell_escape() -> None:
    assert escape_for_bash_lc("echo it's fine") == "echo it'\\''s fine"


def _make_command(
    *,
    name: str = "lint",
    command: str = "uvx ruff check .",
    kind: CommandKind = CommandKind.LINT,
    timeout: int = 120,
    allow_fail: bool = False,
) -> ValidationCommand:
    return ValidationCommand(
        name=name,
        command=command,
        kind=kind,
        timeout=timeout,
        allow_fail=allow_fail,
    )


PLAN_WORKED_EXAMPLE = (
    "__mala_log=/tmp/mala-validation-logs/bd-mala-abc.lint.log\n"
    "mkdir -p /tmp/mala-validation-logs\n"
    "(\n"
    "  if timeout 120 bash -lc 'uvx ruff check .'"
    " >/tmp/mala-validation-logs/bd-mala-abc.lint.log 2>&1; then\n"
    "    __mala_status=0\n"
    "  else\n"
    "    __mala_status=$?\n"
    "  fi\n"
    "  printf 'MALA_EVIDENCE name=%s exit=%s log=%s\\n'"
    " 'lint'"
    ' "$__mala_status" /tmp/mala-validation-logs/bd-mala-abc.lint.log\n'
    '  exit "$__mala_status"\n'
    ")"
)


def test_build_canonical_wrapper_matches_canonical_shape() -> None:
    """Wrapper creates log directory and emits the canonical evidence line."""
    cmd = _make_command()
    actual = build_canonical_wrapper(
        cmd,
        issue_id="bd-mala-abc",
        validation_log_dir=Path("/tmp/mala-validation-logs"),
    )
    assert actual == PLAN_WORKED_EXAMPLE


def test_build_canonical_wrapper_strict_uses_propagating_exit() -> None:
    cmd = _make_command(allow_fail=False)
    wrapper = build_canonical_wrapper(
        cmd,
        issue_id="bd-mala-abc",
        validation_log_dir=Path("/tmp/mala-validation-logs"),
    )
    assert '\n  exit "$__mala_status"\n' in wrapper
    assert "\n  exit 0\n" not in wrapper


def test_build_canonical_wrapper_advisory_uses_zero_exit() -> None:
    cmd = _make_command(allow_fail=True)
    wrapper = build_canonical_wrapper(
        cmd,
        issue_id="bd-mala-abc",
        validation_log_dir=Path("/tmp/mala-validation-logs"),
    )
    assert "\n  exit 0\n" in wrapper
    assert '\n  exit "$__mala_status"\n' not in wrapper


def test_build_canonical_wrapper_log_path_is_absolute_and_composed() -> None:
    cmd = _make_command(name="python_test")
    wrapper = build_canonical_wrapper(
        cmd,
        issue_id="bd-mala-3gbpn.1",
        validation_log_dir=Path("/var") / "log" / "mala",
    )
    assert (
        wrapper.splitlines()[0]
        == "__mala_log=/var/log/mala/bd-mala-3gbpn.1.python_test.log"
    )


def test_build_canonical_wrapper_shell_quotes_log_paths_with_metachars() -> None:
    """The assignment and path uses are safe when log dirs need shell quoting."""
    cmd = _make_command()
    wrapper = build_canonical_wrapper(
        cmd,
        issue_id="bd-mala-abc",
        validation_log_dir=Path("/tmp") / "mala logs-$(touch nope)'x",
    )

    first, mkdir_line, _, redirect_line, *_ = wrapper.splitlines()
    assert first == "__mala_log='/tmp/mala logs-$(touch nope)'\"'\"'x/bd-mala-abc.lint.log'"
    assert mkdir_line == "mkdir -p '/tmp/mala logs-$(touch nope)'\"'\"'x'"
    assert redirect_line.endswith(
        " >'/tmp/mala logs-$(touch nope)'\"'\"'x/bd-mala-abc.lint.log' 2>&1; then"
    )


def test_build_canonical_wrapper_embeds_escaped_command_for_round_trip() -> None:
    """Same-helper round-trip: escape_for_bash_lc(cmd.command) is a literal substring."""
    cmd = _make_command(command='bash -c "echo it\'s fine"')
    wrapper = build_canonical_wrapper(
        cmd,
        issue_id="bd-mala-abc",
        validation_log_dir=Path("/tmp/mala-validation-logs"),
    )
    assert escape_for_bash_lc(cmd.command) in wrapper


def test_build_canonical_wrapper_preserves_configured_timeout() -> None:
    cmd = _make_command(timeout=600)
    wrapper = build_canonical_wrapper(
        cmd,
        issue_id="bd-mala-abc",
        validation_log_dir=Path("/tmp/mala-validation-logs"),
    )
    assert "if timeout 600 bash -lc " in wrapper


def test_build_canonical_wrapper_emits_summary_line_with_command_name() -> None:
    cmd = _make_command(name="security-scan", command="trivy fs .")
    wrapper = build_canonical_wrapper(
        cmd,
        issue_id="bd-mala-abc",
        validation_log_dir=Path("/tmp/mala-validation-logs"),
    )
    assert (
        "  printf 'MALA_EVIDENCE name=%s exit=%s log=%s\\n'"
        " 'security-scan'"
        ' "$__mala_status" /tmp/mala-validation-logs/bd-mala-abc.security-scan.log\n'
    ) in wrapper
